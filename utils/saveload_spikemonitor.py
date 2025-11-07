import numpy as np
from brian2.units import second
from brian2.units import volt
import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
# helper functions to save and load spike montior objects and state monitor objects
# mimic structure of the brian2 objects
# use similar classes for experimental and simulated data, so that we can use the same functions


##### simulation spike monitors ####
sim_path='sim_results/'
print('saving/loading sims from: ',sim_path)

class Spikemonitor_alias:
    def __init__(self, times,times_,indices):
        self.t = times
        self.t_= times_
        self.i = indices

def save_SpM(SpM,fname):
    with open(sim_path+'SpMs/'+fname+'_SpM.npy', 'wb') as f:
        t_= np.save(f, SpM.t_)
        i = np.save(f, SpM.i)

def load_SpM(fname):
    with open(sim_path+'SpMs/'+fname+'_SpM.npy', 'rb') as f:
        t_= np.load(f)
        i = np.load(f)
    SpM=Spikemonitor_alias(t_*second,t_,i)
    return SpM


##### simulation state monitors ####
class Statemonitor_alias:
    def __init__(self, times,times_,v,v_):
        self.t = times
        self.t_= times_
        self.v = v
        self.v_= v_

def save_StM(StM,fname):
    with open(sim_path+'StMs/'+fname+'_StM.npy', 'wb') as f:
        t_= np.save(f, StM.t_)
        v_ = np.save(f, StM.v_)

def load_StM(fname):
    with open(sim_path+'StMs/'+fname+'_StM.npy', 'rb') as f:
        t_= np.load(f)
        v_ = np.load(f)
    StM=Statemonitor_alias(t_*second,t_,v_*volt,v_)
    return StM

def save_StM_withgates(StM,fname):
    with open(sim_path+'StMs/'+fname+'_StM.npy', 'wb') as f:
        t_= np.save(f, StM.t_)
        v_ = np.save(f, StM.v_)
        b = np.save(f, StM.b)
        h = np.save(f, StM.h)

def load_StM_withgates(fname):
    with open(sim_path+'StMs/'+fname+'_StM.npy', 'rb') as f:
        t_= np.load(f)
        v_ = np.load(f)
        b = np.load(f)
        h = np.load(f)
        v=v_*volt
    StM=Statemonitor_alias(t_*second,t_,v,v_)
    StM.b=b
    StM.h=h
    return StM



####  experimental spike monitors #####
class monitor_exp:
    def __init__(self,indices_or,times_or,wb_or):
        self.t_or = times_or
        self.i_or = indices_or
        self.wb_or = wb_or


def load_SpE(path,fname):
    with open(path+fname, 'rb') as f:
        t_or = np.load(f)
        i_or = np.load(f)
        wb_or= np.load(f)
    SpE=monitor_exp(i_or,t_or,wb_or)
    SpE.fname=fname
    return SpE
