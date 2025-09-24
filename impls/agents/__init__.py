from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.tmd import TMDAgent
from agents.fqltmd import FQLTMDAgent
from agents.mdw import MDWAgent
from agents.cmd import CMDAgent
agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    tmd=TMDAgent,
    fqltmd=FQLTMDAgent,
    mdw=MDWAgent,

)
