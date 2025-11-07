import numpy as np
import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from brian2 import * # type: ignore
import utils.brianutils
from utils.brianutils import units
import matplotlib.image as mpimg
import json
import copy
from scipy.interpolate import interp1d
from utils.saveload_spikemonitor import *



# some helper functions for general analysis
def meanfreq2(SpM,dur,tmin=1,i=0):
    """
    calculate mean frequency of unperturbed neurons using the last ISI in simulation
    """
    spikes=SpM.t_[SpM.i==i]
    if len(spikes)>1:
        f=1/np.diff(spikes)[-1]*Hz
    else:
        f=0
    return f

def get_tminmax(SpM):
    """
    gets the first and last point in time for which the phases of all neurons in a spike monitor are defined.
    this corresponds to:
        tmin: first spike of last neuron to start spiking
        tmax: last spike of first neuron to stop spiking
    """
    idx = np.unique(SpM.i)
    tmin=np.max([np.min(SpM.t_[SpM.i==i]) for i in idx])
    tmax=np.min([np.max(SpM.t_[SpM.i==i]) for i in idx])
    return tmin, tmax


def calculate_clean_phases(SpE,dt=0.001):
    """
    calculate phases for experimental recordings as specified in the supplmentary materials.

    - Time is discretized with a time step given by dt.
    - start is the first spike of the last neuron to start spiking
    - end is the last spike of first neuron to stop spiking
    - breaks (where fly stops flying) are excluded.
    - Phases are linearly interpolated between spikes.
    - also calculates ISIs that correspond to the cleaned phases

    """

    # extract
    wb = SpE.wb_or
    spikes=SpE.t_or
    idx=SpE.i_or
    idx_inrecording=np.unique(idx)


    # detect breaks when the fly started/stopped
    iwbis=np.diff(wb)
    wbithresh=0.06 #threshold to detect stopped flying
    break_starts=wb[:-1][iwbis>wbithresh]
    break_ends=wb[1:][iwbis>wbithresh]

    # get tmin and tmax
    tmin = np.max([np.min(spikes[idx==i]) for i in idx_inrecording])
    tmax = np.min([np.max(spikes[idx==i]) for i in idx_inrecording])

    # interpolate phases from tmin to tmax
    φs = interpolate_phases(SpE,tmin,tmax,dt,experimental=True)
    ts = np.arange(tmin,tmax,dt)
    i_boolean=np.ones(ts.size,dtype='bool')

    # exclude phases during breaks
    for b_start,b_end in zip(break_starts,break_ends): #go trhough all the bad intervals
        good_end = np.min([np.max(spikes[(idx==i) & (spikes<b_start)]) for i in idx_inrecording])
        good_start = np.max([np.min(spikes[(idx==i) & (spikes>b_end)]) for i in idx_inrecording])
        i_boolean[(ts>good_end) & (ts<good_start)]=0

    φs_clean = φs[:,i_boolean]


    # also calculate
    ISIs_clean=[]
    for i in idx_inrecording:
        ISIs=np.diff(spikes[idx==i])
        Istart=spikes[idx==i][:-1]
        Iend=spikes[idx==i][1:]

        j_bool=np.ones(ISIs.size,dtype='bool')

        for b_start,b_end in zip(break_starts,break_ends):
            j_bool[(Istart<b_start) & (Iend>b_start)]=0
        ISIs[~j_bool]= np.nan
        ISIs_clean.append(ISIs)

    return φs_clean,φs,i_boolean,ISIs_clean




def interpolate_phases(SpM,tmin,tmax,dt,experimental=False):
    '''
    interpolate phases using spiketimes

    Inputs:
        SpM: spiketime monitor
        tmin: start of interpolation
        tmax: end of interpolation
        dt: time step of interpolation
        experimental: whether SpM is an experimental recording or not

    '''
    if experimental:
        idx_inrecording=np.unique(SpM.i_or)
        spikesls=[np.array(SpM.t_or[SpM.i_or==i]) for i in idx_inrecording]

    else:
        idx_inrecording=np.unique(SpM.i)
        spikesls=[np.array(SpM.t_[SpM.i==i]) for i in idx_inrecording]

    ts=np.arange(tmin,tmax,dt)
    N=idx_inrecording.size
    tsφ=ts[(ts>=tmin) & (ts<=tmax)]

    φs=np.empty([N,tsφ.size])
    for i in range(N):
        spikes=spikesls[i]
        ts2=np.sort(np.concatenate([spikes,spikes+1e-9]))
        spikes2=np.tile(np.array([1,0]),spikes.size)
        φs[i,:]=interp1d(ts2,spikes2)(tsφ)
    return φs


def phase_differences(φs):
    '''check that system has stabilised and then detect whether it is
    in splay in-phase-sync or neither

    INPUTS:
    φs: array of phases

    OUTPUTS:
    differences (string)
    '''
    φs_sorted=np.sort(φs,axis=0)
    diffs=np.vstack([np.diff(φs_sorted,axis=0),(φs_sorted[0,:]-φs_sorted[-1,:])%1])
    return diffs

def order_parameter_r(v,n=1):
    """returns Kuramoto order parameter r, which captures the degree of synchrony"""
    Z=np.mean(np.exp(v*2*np.pi*1j*n),axis=0)
    return np.abs(Z)

def get_splayness(SpM,tmin,tmax,dt,N=5):
    """
    splayness as described in the methods section

    inputs:
        SpM: spiketime monitor
        tmin: start of phase interpolation
        tmax: end of phase interpolation
        dt: time step of phase interpolation

    outputs:
        st: splayness time series
        s: splayness index
    """

    #1: get phases
    φs = interpolate_phases(SpM,tmin,tmax,dt)

    #2: order phases
    φs = np.sort(φs, axis=0)

    #3: get phase differences
    ψs = np.zeros(φs.shape)
    for i in range(N-1):
        ψs[i,:] = (φs[i+1,:]-φs[i,:])
    ψs[-1,:]=1-np.sum(ψs,axis=0)

    #splayness error
    t1=np.sum((ψs-1/N)**2,axis=0)
    t2=(N-1)/N**2 + (1-1/N)**2
    rt=t1/t2

    #splayness
    γt= 1-np.sqrt(rt)
    s = 1-np.sqrt(np.mean(rt))

    return γt,s


def get_splayness_fromphase(φs,dt,N=5):
    """
    Instead of using spiketime monitor, here calculate splayness from phase directly.

    inputs:
        φs: array of phases

    outputs:
        st: splayness time series
        s: splayness index
    """

    #2: order phases
    φs = np.sort(φs, axis=0)

    #3: get phase differences
    ψs = np.zeros(φs.shape)
    for i in range(N-1):
        ψs[i,:] = (φs[i+1,:]-φs[i,:])
    ψs[-1,:]=1-np.sum(ψs,axis=0)

    #splayness error
    t1=np.sum((ψs-1/N)**2,axis=0)
    t2=(N-1)/N**2 + (1-1/N)**2
    rt=t1/t2

    #splayness
    γt= 1-np.sqrt(rt)
    s = 1-np.sqrt(np.mean(rt))

    return γt,s


def scramble_SpM_randomstart(SpM,N=5,exclude05=True,experimental=False):
    """
    shuffle the ISIs in a spike monitor, for calculation of random splayness etc
    start is a random fraction [0,1] of mean ISI in recording.

    inputs:
        SpM: spiketime monitor
        N: number of neurons in recording
        exclude05: whether to exclude ISIs longer than 0.5 seconds
        experimental: whether SpM is an experimental recording

    outputs:
        new spiketime monitor object with shuffled ISIs
    """
    new_ts=[]
    new_is=[]
    for i in range(N):
        if experimental is False:
            ISIs=np.diff(SpM.t_[SpM.i==0])
        if experimental:
            ISIs=np.diff(SpM.t_or[SpM.i_or==0])
        if exclude05:
            ISIs=ISIs[ISIs<0.5]

        np.random.shuffle(ISIs)
        start = np.random.rand()*ISIs.mean()
        new_ts_i=np.cumsum(ISIs)+start
        new_ts.append(new_ts_i)
        new_is.append(np.ones(new_ts_i.size)*i)

    idx=np.argsort(np.concatenate(new_ts))
    new_ts=np.concatenate(new_ts)[idx]
    new_is=np.concatenate(new_is)[idx]

    return Spikemonitor_alias(new_ts*second,new_ts,new_is)
