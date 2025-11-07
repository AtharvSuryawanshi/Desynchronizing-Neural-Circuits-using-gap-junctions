import numpy as np
import sys, os
if os.getcwd() not in sys.path:  
    sys.path.append(os.getcwd())
from utils.brianutils import units
from utils.sim import *
from utils.spiketime_analysis_helper import *
from scipy.interpolate import interp1d



# %% setup getting spike shape

def get_spikeshape(model_dict,dur,threshold,dt):
    """
    run a simple simulation to extract the spike shape.

    inputs:
        model_dict: contains equation and parameters
        dur: duration of simulations
        threshold: threshold condition to detect spikes
        dt: time step of simulation

    output:
        f: frequency of unperturbed neuron
        tspike: array of time points for one spiking period
        vspike: array of voltage values for one period

    """
    StM1, SpM1 = run_sim(1, model_dict,None,None, dur=dur, threshold=threshold, solver='rk4',
                         dt=dt,recordstates=True,statestorecord=['v'],staterecorddt=0.1*ms)

    # take last 3 spikes
    tmin = SpM1.t_[-2]
    tmax = SpM1.t_[-1]
    ts = StM1.t_
    m = (ts>tmin) & (ts<tmax)

    tspike = ts[m]
    vspike = StM1.v[0][m]
    f = meanfreq2(SpM1, dur, tmin=1)

    return f, tspike, vspike



# %% setup getting coupling function

def Γ(ψs,PRC,vspike,ggap,Cm):
    """
    Uses formulas from supplement to calculate

    inputs:
        Δφs: vector of phase differences on which we want to evaluate G
        PRC: PRC from our perturbation method
        vpike: our spike shape

    output:
        Gs: coupling function G
    """
    dψ=np.diff(ψs)[0]
    Gs=np.zeros(ψs.size)*Hz

    for i,ψ in enumerate(ψs):
        ψi=i
        perturbation=pert_1period(ψi,vspike,ggap,Cm)
        Gs[i]=np.sum(PRC*perturbation)*dψ
    return Gs


def pert_1period(δφi,vspike,ggap,Cm):
    """
    perturbation function: Calculates voltage difference that drive gap junction currents
    from 1 neuron to another over 1 period

    inputs:
        vpike: spike shape
        δφi: index of difference between phases
    """
    #voltage that drives gap junction current to neuron 1
    v1=vspike
    v2=np.roll(vspike,δφi) #φ1=φ2+δφ => φ2=φ1-δφ
    return ggap*(v2-v1)/Cm
