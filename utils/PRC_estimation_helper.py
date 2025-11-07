import numpy as np
import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.brianutils import units
from utils.sim import *
from  numpy.linalg import lstsq
from scipy.interpolate import interp1d



def gen_stimtimes(f, dur, num=10):
    """
    generate stimulation times for perturbation based PRC estimation

    Inputs:
        f: mean firing rate of neuron
        dur: duration of PRC measurement

    Outputs:
        stimtimes: random perturbation times
    """
    stims=(np.ones(int(dur*f/2)+num)*1/f*2)
    stims=stims*(np.random.rand(stims.size)*0.5+0.75)
    stimtimes=np.cumsum(stims)
    return stimtimes


def process_spikes(stimtimes,spikes):
    """
    helper function that brings spiketime data into format that can be used
    for implementations of different perturbation based PRC estimation methods

    outputs:
        ISIs0: ISIs during which no perturbation occured
        ISIs1: ISIS during which 1 perturbation occured
        tjs: relative time points of perturbation since beginning of ISI1
    """


    ISIs=np.diff(spikes)
    stimtimes=stimtimes[(stimtimes>spikes[0]) & (stimtimes<spikes[-1])] #ignore stims before and after spikes

    _,i,_=np.intersect1d(stimtimes,spikes,return_indices=True)
    stimtimes=np.delete(stimtimes,i) #ignore stims on spike time
    idx=np.digitize(stimtimes,spikes)-1 #index over the ISIs


    u,i,c=np.unique(idx,return_index=True,return_counts=True)

    idx1=idx[i[(c==1)]]
    stimtimes1=stimtimes[i[(c==1)]]
    tjs=stimtimes1-spikes[idx1]
    ISIs1=ISIs[idx1]

    ISIs0=np.delete(ISIs,idx)

    return ISIs0,ISIs1,tjs





def calc_PRC(ISIs1,tjs,T,vs,plot=True):
    """
    construct PRC based on results of perturbation simulation.
    This is vaguely based on  Galan et al. (2005).
    However, whereas Galan et al. (2005) use a Fourier basis, this is not well-suited
    to represent the high frequencies in e.g. SNL PRC.

    Warning: Some details are not well defined in the paper, e.g. what happens
    with stimtimes that occur after T. I made some decisions, but other
    implementations might differ. Different implementations may therefore
    lead to slightly different results.

    Inputs:
        T: mean firing period
        vs: array of phases over which to compute PRC

    Outputs:
        zs: PRC values at phases given by vs
    """


    #setup
    Δφs=1-ISIs1/T #phase deviations
    φps=tjs/T #phases at which current pulse occurs

    Δφs=Δφs[φps<1] # not defined in Galan, we delete these cases
    φps=φps[φps<1] 


    def PRC_interp(φs,φps=φps,Δφs=Δφs):
            xvals=np.hstack([φps-1,φps,φps+1])
            yvals=np.hstack([Δφs,Δφs,Δφs])
            return interp1d(xvals,yvals,bounds_error=True)(φs%1).flatten()

    PRC=PRC_interp(vs)
    return PRC



def normalize_PRC(PRC):
    """
    normalize PRC to 1
    """
    PRC_norm=PRC/np.abs(PRC).max()
    return PRC_norm





def PRC_sim(model_dict,stimtimes,dur=20*second,ε=0.5*mV,threshold="v>-10*mV",solver='rk4',dtsim=20*us):
    """
    simulation with perturbations
    """
    md = copy.deepcopy(model_dict)
    neuron_equation= brianutils.load_model(md)

    start_scope()

    G1= NeuronGroup(1, model=neuron_equation, method=solver,dt=dtsim,
                                    refractory="5*ms", threshold=threshold)

    G2 = SpikeGeneratorGroup(1,np.zeros(stimtimes.shape),stimtimes,dt=dtsim)
    eqs2 = '''w: volt
                 '''
    S1=Synapses(G2, G1, eqs2, on_pre='v_post += w',dt=dtsim)
    S1.connect()
    S1.w=ε

    StM = StateMonitor(G1, ['v'], record=True)
    SpM = SpikeMonitor(G1)

    net= Network(G1,G2,S1,StM,SpM)
    net.run(dur,report='text')

    spikes=SpM.t
    spikes=spikes[3:]

    return spikes
