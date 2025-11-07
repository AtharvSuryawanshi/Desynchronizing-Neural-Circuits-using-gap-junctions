import numpy as np
from brian2 import *
import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import utils.brianutils
from utils.brianutils import units
from utils.saveload_spikemonitor import *





# random initial conditions SNL case
N=1
v0=-60*mV
fname='initial_SNL'

SpM_init=load_SpM(fname)
StM_init=load_StM_withgates(fname)

def random_initial_phase_SNL(N,StM_init=StM_init,SpM_init=SpM_init,replace=True):
    """
    uses simulation of unperturbed SNL neuron to sample random intial condition.
    """

    ts=StM_init.t_
    tmin=SpM_init.t_[-2]
    tmax=SpM_init.t_[-1]
    m=(ts>tmin) & (ts<tmax)

    vs=StM_init.v[0,m]
    bs=StM_init.b[0,m]
    hs=StM_init.h[0,m]

    idx=np.random.choice(np.arange(vs.size),size=N,replace=replace)

    initial_values={'v':vs[idx],'b':bs[idx],'h':hs[idx]}
    return initial_values


def pick_initial_phase_SNL(φs_chosen,StM_init=StM_init,SpM_init=SpM_init):
    """
    nonrandom equivalent: set the phase yourself
    """
    N=len(φs_chosen)
    ts=StM_init.t_
    tmin=SpM_init.t_[-2]
    tmax=SpM_init.t_[-1]
    m=(ts>tmin) & (ts<tmax)

    vs=StM_init.v[0,m]
    bs=StM_init.b[0,m]
    hs=StM_init.h[0,m]

    idx=np.array(np.around(vs.size*φs_chosen),dtype='int')

    initial_values={'v':vs[idx],'b':bs[idx],'h':hs[idx]}
    return initial_values



# random initial conditions SNIC case
N=1
v0=-60*mV
fname='initial_SNIC'

SpM_init=load_SpM(fname)
StM_init=load_StM_withgates(fname)

def random_initial_phase_SNIC(N,StM_init=StM_init,SpM_init=SpM_init,replace=True):
    """
    uses simulation of unperturbed SNIC neuron to sample random intial condition.
    """

    ts=StM_init.t_
    tmin=SpM_init.t_[-2]
    tmax=SpM_init.t_[-1]
    m=(ts>tmin) & (ts<tmax)

    vs=StM_init.v[0,m]
    bs=StM_init.b[0,m]
    hs=StM_init.h[0,m]

    idx=np.random.choice(np.arange(vs.size),size=N,replace=replace)
    initial_values={'v':vs[idx],'b':bs[idx],'h':hs[idx]}
    return initial_values
