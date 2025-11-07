import numpy as np
from brian2.units import nS,nsiemens
from scipy.interpolate import interp1d
import json
import sys, os
if os.getcwd() not in sys.path:  
    sys.path.append(os.getcwd())


#load analytical expressions
cc_str_expr=json.load(open('cfg/cc_str_expr.json'))
str_cc_ana=cc_str_expr['str_cc_ana']
str_ggap_ana=cc_str_expr['str_ggap_ana']

def cc2ggap(cc):
    """
    convert copuling coefficient to coupling strength
    """
    return eval(str(str_ggap_ana))

def ggap2cc(ggap):
    """
    convert coupling strength to coupling coefficient
    """
    return eval(str(str_cc_ana))
