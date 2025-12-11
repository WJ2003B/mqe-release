from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    DiscreteStateActionRepresentation,
    StateRepresentation,
    ActorVectorField
)

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
# from utils.networks import ActorVectorField


class FQLTMDAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with TMD as Q values."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def mrn_distance(self, x, y):
        
        K = self.config['components']
        assert x.shape[-1] % K == 0
        # K = 8
        # @jax.jit
        # def mrn_distance_component(x, y):
        #     eps = 1e-6
        #     max_component = jnp.max(jax.nn.relu((x - y)), axis=-1)
        #     return max_component #+ l2_component

        x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
        fn = lambda x, y: jnp.max(jax.nn.relu((x - y)), axis=-1)
        dists = jax.vmap(fn, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)
        # print(dists.shape)
        #[self.mrn_distance_component(x_split[..., i], y_split[..., i]) for i in range(K)]

        return dists.mean(axis=-1)
    
    @jax.jit
    def get_phi(self, phi_, psi_s):
        return psi_s + jax.nn.relu(phi_)

    def critic_loss(self, batch, grad_params, step):
        """Compute the FQL critic loss."""
        batch_size = batch['observations'].shape[0]

        phi_ = self.network.select('phi')(batch['observations'], batch['actions'], params=grad_params)
        # concat_obs = jnp.concatenate([batch['observations'], batch['next_observations'], batch['value_goals']], axis=0)
        # reps = self.network.select('psi')(concat_obs, params=grad_params)
        # if len(reps.shape) == 2:
        #     reps = reps[None, ...]
        # psi_s, psi_next, psi_g = jnp.split(reps, 3, axis=1)
        psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
        psi_next = self.network.select('psi')(batch['next_observations'], params=grad_params)
        psi_g = self.network.select('psi')(batch['value_goals'], params=grad_params)

        phi = self.get_phi(phi_, psi_s)

        if not self.config['ensemble']:  # Non-ensemble
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]

        dist = self.mrn_distance(phi[:, :, None], psi_g[:, None, :])
        logits = -dist / jnp.sqrt(phi.shape[-1])
        # logits.shape is (e, B, B) with one term for positive pair and (B - 1) terms for negative pairs in each row.

        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        action_dist = self.mrn_distance(psi_s, phi)

        action_invariance_loss = jnp.mean(action_dist)

        dist_next = self.mrn_distance(psi_next[:, :, None], psi_g[:, None, :])

        t = self.config['t']
        gamma = self.config['discount']
        dist_next = jax.lax.stop_gradient(dist_next)

        delta = dist - dist_next
        mask = delta > t
        delta_clipped = jnp.where(mask, t, delta)
        # next_prob = jnp.exp(-dist_next)
        # divergence_clipped = next_prob * (jnp.log(gamma) + dist) - (1 - next_prob) * jnp.log(1 - jnp.exp(-dist) * gamma)
        divergence = jnp.where(mask, delta, gamma * jnp.exp(delta_clipped) - dist)


        dw = self.config['diag_backup']
        divergence = divergence * (1 - dw) + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw

        # divergence = jnp.clip(divergence, -self.config['t'], self.config['t'])
        backup_loss = jnp.mean(divergence)
        # zeta = self.zeta_schedule(step)
        zeta=self.config["zeta"]

        critic_loss = (
            contrastive_loss
            # + zeta * action_invariance_loss
            + zeta * backup_loss
        )

        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, {
            'critic_loss': critic_loss,
            'action_invariance_loss': action_invariance_loss,
            'backup_loss': backup_loss,
            'contrastive_loss': contrastive_loss,
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
            'dist': dist.mean(),
            'biggest_diff_in_dist': jnp.max(dist - dist_next),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        # else:
        if self.config['use_latent']:
            psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
            psi_g = self.network.select('psi')(batch['actor_goals'], params=grad_params)
            if len(psi_g.shape) == 3:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            if self.config['freeze_enc_for_actor_grad']:
                psi_g = jax.lax.stop_gradient(psi_g)
                psi_s = jax.lax.stop_gradient(psi_s)
            pred = self.network.select('actor_bc_flow')(jax.lax.stop_gradient(psi_s), x_t, jax.lax.stop_gradient(psi_g), t, params=grad_params)
            bc_flow_loss = jnp.mean((pred - vel) ** 2)

            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(psi_s, psi_g, noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(psi_s, noises, psi_g, params=grad_params)
        else:
            pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, batch["actor_goals"], t, params=grad_params)
            bc_flow_loss = jnp.mean((pred - vel) ** 2)
            # Distillation loss.
            rng, a_rng = jax.random.split(rng)
            noises = jax.random.normal(a_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], batch["actor_goals"], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'],  noises, batch["actor_goals"], params=grad_params)
        
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        phi_ = self.network.select('phi')(batch['observations'], actor_actions)
        psi_s = self.network.select('psi')(batch['observations'])
        psi_g = self.network.select('psi')(batch['actor_goals'])
        phi = self.get_phi(phi_, psi_s)
        q1, q2 = -self.mrn_distance(phi, psi_g)
        q = jnp.minimum(q1, q2)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], batch['actor_goals'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, step=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, step)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info


    @jax.jit
    def update(self, batch, contrastive_only=False, step=None):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        if self.config['use_latent']:
            psi_s = self.network.select('psi')(observations)
            psi_g = self.network.select('psi')(goals)
            if psi_g.shape[0] == 2:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            actions = self.network.select('actor_onestep_flow')(psi_s, noises, psi_g)
        else:
            actions = self.network.select('actor_onestep_flow')(observations, noises, goals)
        actions = jnp.clip(actions, -1, 1)
        return actions


    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        goals,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, goals, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def get_distance(self, observations, goals, actions):
        #actions not used, will be used for cmd
        if self.config['use_action_for_distance']:
            phi = self.network.select('phi')(observations, actions)
        else:
            phi = self.network.select('psi')(observations)
        psi = self.network.select('psi')(goals)
        return self.mrn_distance(phi, psi)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = jnp.zeros_like(ex_observations)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]


        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['phi'] = encoder_module()
            encoders['psi'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        if config['discrete']:
            phi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=config['ensemble'],
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
            psi_def = DiscreteStateActionRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=config['ensemble'],
                value_exp=True,
                state_encoder=encoders.get('state'),
                action_dim=action_dim,
            )
        else:
            phi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=config['ensemble'],
                value_exp=True,
                state_encoder=encoders.get('state'),
            )
            psi_def = StateRepresentation(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=config['ensemble'],
                value_exp=True,
                state_encoder=encoders.get('state'),
            )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )
        if not config['use_latent']:
            network_info = dict(
                phi=(phi_def, (ex_observations, ex_actions)),
                psi=(copy.deepcopy(psi_def), (ex_observations, )),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_goals, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions, ex_goals)),)
        else:
            embed = jnp.zeros((1, config['latent_dim']))
            network_info = dict(
                phi=(phi_def, (ex_observations, ex_actions)),
                psi=(copy.deepcopy(psi_def), (ex_observations, )),
            actor_bc_flow=(actor_bc_flow_def, (embed, ex_actions, embed, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (embed, ex_actions, embed)),)
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fqltmd',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            ensemble=True, # Whether to ensemblize
            components=8,  # Number of components in MRN.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            # tau=0.005,  # Target network update rate.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            zeta=0.01,
            t=3.0,
            diag_backup=0.5,  # Whether to use diagonal backup.
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
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            use_iqe=False,  # Whether to use IQE distance or MRN distance
            use_latent=False, # Whether to use latent for policy action sampling
            freeze_enc_for_actor_grad=False, # Whether to stop grad for actor when using encoder
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            use_action_for_distance=False, # Whether to use action for distance calculation
            direct_optimization=True, # Whether to directly optimize the actor
        )
    )
    return config
