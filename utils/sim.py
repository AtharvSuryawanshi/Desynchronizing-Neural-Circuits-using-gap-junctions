import numpy as np
from brian2 import *
import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils import brianutils
from utils.brianutils import units
import json
import copy


def run_sim(N,model_dict,g_gap,wsyn,dur=2*second,I_in=None,threshold="v>-10*mV",
            solver='rk4',dt=10*us,σnoise=None,initial_values=None,v0=None,
            recordstates=True,statestorecord=['v'],staterecorddt=0.1*ms):
    """
    Main simulation function

    Parameters
    ----------
    N:
        number of neurons in Network
    model_dict:
        contains
    g_gap:  brian2 quantity or array [in Siemens]
        gap junction coupling strength
    wsyn:
        for chemical synapses (not used in this paper)
    dur:
        duration of simulation [seconds]
    I_in:
        set input current. If None, defaults to value in model_dict
    threshold:
        threshold condition to detect spikes
    solver:
        numerical intergration method
    σnoise:
        noise strength. If not none, solver has to be 'heun'


    Returns
    -------
    StM1 : StateMonitor object
        contains variables specified by statestorecord
    SpM1 : SpikeMonitor object
        contains spike times of neurons in network

    """

    #create neuron_equation
    md = copy.deepcopy(model_dict)

    if σnoise is not None:
        md['parameters']['sig']='{} * second**0.5 *uA'.format(σnoise)
        md['ode'][0]=md['ode'][0].replace('I_in','I_in + sig * xi')
        assert solver=='heun'

    if I_in is not None:
        neuron_equation= brianutils.load_model(md,["I_gap","I_in"])
    else:
        neuron_equation= brianutils.load_model(md,["I_gap"])


    # define neurons and monitors
    G = NeuronGroup(N, neuron_equation,method=solver,threshold=threshold,refractory='10*ms',dt=dt)

    net= Network(G)

    SpM1 = SpikeMonitor(G)
    net.add(SpM1)

    if recordstates:
        StM1 = StateMonitor(G, statestorecord, record=True,dt=staterecorddt)
        net.add(StM1)

    # add electrical synapses
    if g_gap is not None:
        syn_eqs='''I_gap_post= g_gap * (v_pre - v_post) : amp (summed)
                g_gap: siemens'''
        S = Synapses(G, G, syn_eqs)

        if type(g_gap/uS) is float:
            S.connect(condition='i != j')
            S.g_gap=g_gap
        elif type(g_gap/uS) is numpy.ndarray:
            S.connect()
            S.g_gap=g_gap.flatten()
        net.add(S)

    # add chemical synapses
    if wsyn is not None:
        on_pre = 'I_syn_post += {}*nA'.format(wsyn)
        S2 = Synapses(G, G, on_pre=on_pre)
        S2.connect(condition='i != j')
        net.add(S2)

    #set states
    if initial_values is not None:
        G.set_states(initial_values)
    if v0 is not None:
        G.v=v0
#     G.I_syn=0
    G.I_gap=0
    if I_in is not None:
        G.I_in=I_in

    # run simulation
    net.run(dur,report='text')

    if recordstates:
        return StM1,SpM1
    else:
        return SpM1
