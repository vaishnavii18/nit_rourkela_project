import numpy as np
from config.settings import BANDWIDTH

def distance(p1,p2):
    return np.linalg.norm(p1-p2)

def throughput(sinr):
    return BANDWIDTH*np.log2(1+sinr)

def channel_gain(d):
    return 1/(d**2+1)