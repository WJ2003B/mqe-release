from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import copy

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    DiscreteStateActionRepresentation,
    GCActor,
    GCDiscreteActor,
    Param,
    StateRepresentation,
)


class MQEAgent(flax.struct.PyTreeNode):
    """Metric Distillation via Waypoints (MQE) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def mrn_distance(self, x: jnp.ndarray, y: jnp.ndarray):
        K = self.config['components']
        assert x.shape[-1] % K == 0

        @jax.jit
        def mrn_distance_component(x: jnp.ndarray, y: jnp.ndarray):
            eps = 1e-8
            d = x.shape[-1]
            mask = jnp.arange(d) < d // 2
            max_component: jnp.ndarray = jax.nn.relu(jnp.max((x - y) * mask, axis=-1)) 
            l2_component: jnp.ndarray = jnp.linalg.norm((x - y) * (1 - mask) + eps, axis=-1)
            # assert max_component.shape == l2_component.shape
            return max_component + l2_component

        x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
        dists: jnp.ndarray = jax.vmap(mrn_distance_component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)

        return dists.mean(axis=-1) / jnp.sqrt(x.shape[-1])

    def iqe_distance(self, x, y):
        k = self.config['components']
        alpha_raw = self.network.select('alpha_raw')()
        alpha = jax.nn.sigmoid(alpha_raw)
        reshape = (x.shape[-1] // k, k)
        x = jnp.reshape(x, (*x.shape[:-1], *reshape))
        y = jnp.reshape(y, (*y.shape[:-1], *reshape))
        valid = x < y
        D = x.shape[-1]
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(ixy < D, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = (neg_inp_copies < 0) * (-1.0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(-1)
        result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
        return result

    @jax.jit
    def distance(self, x, y) -> jnp.ndarray:
        # x, y = jnp.broadcast_arrays(x, y)
        if self.config['use_iqe']:
            return self.iqe_distance(x, y)
        else:
            return self.mrn_distance(x, y)


    @jax.jit
    def critic_loss(self, batch, grad_params, critic_rng):
        if self.config["batch_size"] == self.config["critic_batch_size"]:
            batch = batch
            batch_size = self.config['batch_size']
        else:
            key = jax.random.PRNGKey(critic_rng[0])
            batch_size = self.config['critic_batch_size']
            idx2 = jax.random.permutation(key, self.config['batch_size'])[:self.config['critic_batch_size']]
            batch = jax.tree.map(lambda x: x[idx2], batch)
        key = jax.random.PRNGKey(critic_rng[1])
        use_next_state = jax.random.bernoulli(key, p=self.config['next_state_sample'], shape=(batch_size,))
        use_next_state_mask = jnp.reshape(use_next_state, (batch_size, *[1] * (len(batch['observations'].shape) - 1)))
        intermediate_value_goals = jnp.where(use_next_state_mask, batch['next_observations'], batch['intermediate_value_goals'])

        batch_size = batch['observations'].shape[0]
        if self.config['encoder'] is not None:
            phi, phi_ = self.network.select('phi')(batch['observations'], batch['actions'], params=grad_params)
            psi_s, psi_s_ = self.network.select('psi')(batch['observations'], params=grad_params)
            psi_next, _ = self.network.select('psi')(intermediate_value_goals, params=grad_params)
            psi_g, _ = self.network.select('psi')(batch['value_goals'], params=grad_params)
        else:
            phi = self.network.select('phi')(batch['observations'], batch['actions'], params=grad_params)
            psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
            phi_ = phi
            psi_s_ = psi_s
            psi_next = self.network.select('psi')(intermediate_value_goals, params=grad_params)
            psi_g = self.network.select('psi')(batch['value_goals'], params=grad_params)

        if len(psi_s.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]
        
        # phi = self.get_phi(phi_, psi_s)

        # # logits.shape is (e, B, B) with one term for positive pair and (B - 1) terms for negative pairs in each row. 
        dist = self.distance(phi[:, :, None], psi_g[:, None, :]) 
        dist_next = self.distance(psi_next[:, :, None], psi_g[:, None, :])
 
        I = jnp.eye(batch_size)
        logits = -dist # / jnp.sqrt(phi.shape[-1])


        action_dist = self.distance(psi_s, phi)
        # weight_reg = self.config['weight_decay'] * jnp.mean(jnp.square(phi))
        action_invariance_loss = jnp.mean(jnp.square(action_dist))

        # print('dist.shape', dist.shape)
        # print('dist_next.shape', dist_next.shape)

        def compute_backup(dist, dist_next):
            t = self.config['t']
            gamma = self.config['discount']
            delta = dist - dist_next
            mask = delta > t
            delta_clipped = jnp.where(mask, t, delta)
            # use_next_state_mask.squeeze(1) should have shape (B,)
            one_step_mask = jnp.where(use_next_state_mask.reshape(use_next_state_mask.shape[0],), 1.0, batch['intermediate_value_goals_offsets'])[None,:,None]
            # print(delta.shape)
            # print(one_step_mask.shape)

            s = gamma ** one_step_mask
            divergence = jnp.where(mask, delta, s * jnp.exp(delta_clipped) - dist)
            dw = self.config['diag_backup']
            optim_value = 1 - jax.lax.stop_gradient(dist_next) + jnp.log(gamma) * one_step_mask
            optim_value = optim_value * (1 - dw) + jnp.diagonal(optim_value, axis1=1, axis2=2)[..., None] * dw
            diag = jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
            divergence = divergence * (1 - dw) + diag
            optim_backup = jnp.mean(optim_value)
            return jnp.mean(divergence), optim_backup
        backup_loss, optim_backup = compute_backup(dist, jax.lax.stop_gradient(dist_next))
        optim_backup = jnp.mean(optim_backup)

        zeta, backup_weight = 1.0, 1.0
        invariance_weight = self.config['zeta']
        critic_loss = backup_loss + action_invariance_loss
        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return (
            # (critic_loss, backup_loss),
            critic_loss,
            {
                'critic_loss': critic_loss,
                'backup_loss': backup_loss,
                'backup_optim_loss': backup_loss - optim_backup,
                'action_invariance_loss': action_invariance_loss,
                'zeta': zeta,
                'invariance_weight': invariance_weight,
                'backup_weight': backup_weight,
                'binary_accuracy': jnp.mean((logits > 0) == I),
                'categorical_accuracy': jnp.mean(correct),
                'logits_pos': logits_pos,
                'logits_neg': logits_neg,
                'logits': logits.mean(),
                'dist': dist.mean(),
                'phi_mag': jnp.mean(jnp.abs(phi)),
                'psi_s_mag': jnp.mean(jnp.abs(psi_s)),
                # 'phi_relu_mag': jnp.mean(jax.nn.relu(phi_)),
                # 'dist_': dist_.mean(),
                'biggest_diff_in_dist': jnp.max(dist - dist_next),
            },
        )

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):
        # Maximize log Q if actor_log_q is True (which is default).

        if self.config['use_latent']:
            if self.config['freeze_enc_for_actor_grad']:
                # psi_s = self.network.select('psi')(batch['observations'])
                psi_g = self.network.select('psi')(batch['actor_goals'])
                if len(psi_g.shape) == 3:
                    # psi_s = jnp.mean(psi_s, axis=0)
                    psi_g = jnp.mean(psi_g, axis=0)
                dist = self.network.select('actor')(batch['observations'], psi_g, params=grad_params)
                dist_bc = dist
            else:
                # psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
                psi_g = self.network.select('psi')(batch['actor_goals'], params=grad_params)
            
                if len(psi_g.shape) == 3:
                    # psi_s = jnp.mean(psi_s, axis=0)
                    psi_g = jnp.mean(psi_g, axis=0)
                # goals = jnp.concatenate([batch['actor_goals'], psi_g], axis=-1)
                dist = self.network.select('actor')(jax.lax.stop_gradient(batch['observations']), jax.lax.stop_gradient(psi_g), params=grad_params)
                dist_bc = self.network.select('actor')(batch['observations'], psi_g, params=grad_params)
        else:
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            dist_bc = dist
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        if self.config['encoder'] is not None:
            phi, _ = self.network.select('phi')(batch['observations'], q_actions)
            # psi_s = self.network.select('psi')(batch['observations'])
            psi_g, _ = self.network.select('psi')(batch['actor_goals'])
            # phi = self.get_phi(phi_, psi_s)
        else:
            phi = self.network.select('phi')(batch['observations'], q_actions)
            psi_g = self.network.select('psi')(batch['actor_goals'])
        q1, q2 = -self.distance(phi, psi_g)
        q = jnp.minimum(q1, q2)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        if self.config["normalize_q_loss"]:
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        else:
            q_loss = -q.mean()
        log_prob = dist_bc.log_prob(batch['actions'])
        # log_prob_policy = dist.log_prob(batch['actions'])
        # alignment = jnp.minimum(self.config['alignment'], self.config['alpha'])
        # bc_loss = -(alignment * log_prob).mean() - (self.config['alpha'] - alignment) * log_prob_policy.mean()
        bc_loss = -(self.config['alpha'] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @partial(jax.jit, static_argnames="critic_only")
    def total_loss(self, batch, grad_params, rng=None, critic_only=False, step: int=0):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng = jax.random.split(rng)

        # if critic_only:
        #     loss, info_one = self.critic_loss(batch, grad_params, key)
        #     for k, v in info_one.items():
        #         info[f'critic/{k}'] = v
        # else:
        #     loss, info_one = self.actor_loss(batch, grad_params, key)
        #     for k, v in info_one.items():
        #         info[f'actor/{k}'] = v


        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng
        )
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        total_loss = critic_loss + actor_loss
        return total_loss, info

    @partial(jax.jit, static_argnames="critic_only")
    def update(self, batch, critic_only=False, step: int=0):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, critic_only=critic_only, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        if self.config['use_latent']:
            # psi_s = self.network.select('psi')(observations)
            psi_g = self.network.select('psi')(goals)
            if psi_g.shape[0] == 2:
                # psi_s = jnp.mean(psi_s, axis=0)/
                psi_g = jnp.mean(psi_g, axis=0)
            # goals = jnp.concatenate([goals, psi_g], axis=-1)
            dist = self.network.select('actor')(observations, psi_g, temperature=temperature)
        else:
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def get_distance(self, observations, goals, actions):
        #actions not used, will be used for cmd
        if self.config['use_action_for_distance']:
            psi = self.network.select('psi')(observations)
            phi = self.network.select('phi')(observations, actions)
        else:
            phi = self.network.select('psi')(observations)
        psi = self.network.select('psi')(goals)
        return self.distance(phi, psi)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        train_steps
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # zeta_schedule = optax.exponential_decay(
        #     init_value=config['zeta'],
        #     transition_steps=train_steps,
        #     decay_rate=0.999,
        #     end_value=0.0,
        # )
        config['gamma'] = config['discount']

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            if not config['use_latent']:
                encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['state'] = encoder_module()
        if config['discrete']:
            phi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            psi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            phi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                # dropout=config['dropout'],
                state_encoder=encoders.get('state'),
            )
            psi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                # dropout=config['dropout'],
                state_encoder=encoders.get('state'),
            )
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )
        if config['use_iqe']:
            network_info = dict(
                actor=(actor_def, (ex_observations, ex_goals)),
                phi=(phi_def, (ex_observations, ex_actions)),
                psi=(psi_def, (ex_goals,)),
                alpha_raw=(Param(), ()),
            )
        else:
            if config['use_latent']:
                embed = jnp.zeros((1, config['latent_dim']))
                # goals = jnp.concatenate([ex_goals, embed], axis=-1)
                network_info = dict(
                    actor=(actor_def, (ex_observations, embed)),
                    phi=(phi_def, (ex_observations, ex_actions)),
                    psi=(psi_def, (ex_goals,)),
                    backup_coeff=(Param(), ()),
                    invariance_coeff=(Param(), ()),
                )
            else:
                network_info = dict(
                    actor=(actor_def, (ex_observations, ex_goals)),
                    phi=(phi_def, (ex_observations, ex_actions)),
                    psi=(psi_def, (ex_goals,)),
                    backup_coeff=(Param(), ()),
                    invariance_coeff=(Param(), ()),
                )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])# , weight_decay=config['weight_decay'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        # zeta_optimizer = DualOptimizer()
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='mqe',  # Agent name.
            lr=3e-4,
            weight_decay=1e-4, # weight decay for adamw
            components=8,  # Number of components to average in the MRN/IQE distance ensemble.
            batch_size=256,  # Batch size.
            critic_batch_size=256,  # Batch size for critic.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            dropout=0.1, # dropout for just the value network
            discount=0.99,  # Discount factor.
            lambda_=0.99, # lambda for n-step backup
            gamma=0.995,
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            zeta=0.3,  # Weight for TMD backup and invariance losses.
            t=5.0,  # Clipping threshold for the backup LINEX loss.
            diag_backup=0.5,  # Weighting of backups on diagonal (i.e., for s,g ~ p(s,g)) vs. off-diagonal (i.e., for s,g ~ p(s)p(g)).
            stopgrad_psi_backup=False,  # Whether to stop gradient for psi in the backup loss.
            stopgrad_phi_invariance=False,  # Whether to stop gradient for phi in the invariance loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            intermediate_value_geom_sample=True,  # Whether to use geometric sampling for intermediate value goals, otherwise defaults to uniform sampling.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            use_iqe=False,  # Whether to use IQE distance or MRN distance
            use_latent=False,  # Whether to use latent for policy action sampling
            freeze_enc_for_actor_grad=False,  # Whether to stop grad for actor when using encoder
            action_invariance_arch=True,
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            use_action_for_distance=True,  # Whether to use action for distance computation
            log_invariance=False,  # Whether to log invariance loss
            normalize_q_loss=True,  # Whether to normalize Q loss
            next_state_sample=0.2, # probability of using next state as value goal
            cotrain_steps=500_000, # number of steps to cotrain
            alignment=1e-3,  # weight for alignment loss
            simplify_dist=True,  # Whether to simplify distance computation
        )
    )
    return config
