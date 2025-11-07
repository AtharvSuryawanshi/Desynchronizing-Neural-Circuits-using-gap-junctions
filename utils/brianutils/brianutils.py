import brian2

units= dict(
            list(vars(brian2.units).items())
           +list(vars(brian2.units.allunits).items())
           +list(vars(brian2.units.fundamentalunits).items())
           )

def sde2ode(sde):
  """ Converts brian2 stochastic differential equations to ordinary
      differential equations by removing stochastic terms.

      INPUT
        sde : brian2.Equations object
      OUTPUT
        ode : brian2.Equations object
  """
  from sympy import S
  from brian2 import Equations, is_dimensionless
  assert sde.is_stochastic
  odelist = [(i,str(S(j).subs([(k,0) for k in sde.stochastic_variables]))
    + " : " + repr(sde.dimensions[i])) for i,j in sde.eq_expressions] # 
  odelist = [(i,j.replace("Dimension()","1")) for i,j in odelist]
  ode = Equations("d{}/dt = {}".format(*odelist[0]))
  
  for i,j in odelist[1:]:
    ode += Equations("d{}/dt = {}".format(i,j))
  
  for par in sde.parameter_names:
    par_unit = sde.dimensions[par]
    if is_dimensionless(par_unit):
      par_unit = repr(1)
    else:
      par_unit = repr(par_unit)
    ode+=Equations("{0} : {1}".format(par,par_unit))

  return ode

def sde2fluct(sde):
  """ Obtains the diffusion tensor from a brian2 stochastic differential
      equation by gathering the factors in front of xi terms.
  
      INPUT
        sde      : brian2.Equations object
      OUTPUT
        Smat     : sqrt of the diffusion matrix
        stochvar : list of stochastic variables
  """
  var,rhs= zip(*sde.eq_expressions)
  stochvar = list(sde.stochastic_variables)
  return [[sympy.S(j).coeff(k) for k in stochvar] for j in rhs], stochvar

# Old loadmod function for backwards compatibility (used by Janina)
def oldloadmod(modeldict,bifpar,r=4):
  """
  INPUT
    modeldict : dictionary with model secifications
    bifpar    : dictionary or list with bifurcation parameters
    r         : recursion depth for substitution
  OUTPUT
    ode/sde   : brian2.Equations object
  """
  from load_model import rsubs
  import sympy
  # Definitions (of the Ionic currents)
  keylist= ["def","currents","defs"]
  keys= [k for k in keylist if k in modeldict]
  if keys:
    k=keys[0]
    if len(keys)>1:
      raise Warning("Definitions: More than one key {}. Use {}.".format(keys,k))
  else:
    raise KeyError("Definitions: None of the keys {} found.".format(keylist))
  defs= dict(modeldict[k])

  # Functions
  keylist= ["foo","foos","fun","funs"]
  keys= [k for k in keylist if k in modeldict]
  if keys:
    k=keys[0]
    if len(keys)>1:
      warnings.warn("Functions: More than one key {}. Use {}.".format(keys,k))
  else:
    raise KeyError("Functions: None of the keys {} found.".format(keylist))
  foos= dict(modeldict[k])

  # Parameter
  keylist= ["par","pars","params","param"]
  keys= [k for k in keylist if k in modeldict]
  if keys:
    k=keys[0]
    if len(keys)>1:
      warnings.warn("Parameters: More than one key {}. Use {}.".format(keys,k))
  else:
    raise KeyError("Parameters: None of the keys {} found.".format(keylist))
  pars= dict(modeldict[k])
  for k in bifpar:
    if k in pars: 
      pars.pop(k)
    else:
      warnings.warn("bifpar {} not found in parameter dict.".format(k))

  # ODE
  keylist= ["ode","odes","sde","sdes","de","deq","deqs"]
  keys= [k for k in keylist if k in modeldict]
  if keys:
    k=keys[0]
    if len(keys)>1:
      raise Warning("ODEs: More than one key {}. Use {}.".format(keys,k))
  else:
    raise KeyError("ODEs: None of the keys {} found.".format(keylist))
  deq= dict(modeldict[k])
  
  # >>> the following is added this to prevent sympy bug, and to transform, e.g., alpha(v)->alpha: ---------------------------------
  def substituteString (thisString, replacementDict):
    for substitutionString in replacementDict.keys():
      thisString = thisString.replace(substitutionString, replacementDict[substitutionString])
    return thisString

  replacementDict = {}
  for key in foos.keys():
    foos[key] = "({0})".format(foos[key])
    if "(v)" in key:
        replacementDict[key] = key[:key.rfind(('v'))-1]
        foosKeySave = str(foos[key])
        foos.pop(key)
        foos[replacementDict[key]] = foosKeySave
  for key in foos.keys():
    foos[key] = substituteString(foos[key], replacementDict)
  for key in deq.keys():
    deq[key] = substituteString(deq[key], replacementDict)
  for key in defs.keys():
    defs[key] = substituteString(defs[key], replacementDict)
    defs[key] = "({0})".format(defs[key])
  for k in pars:
    pars[k] = "({0})".format(pars[k])
  # <<< ---------------------------------------------------------------------------------------

  for k in ["units","init","varinit"]:
    if k in modeldict:
      varunits= dict([(i,"1") if brian2.is_dimensionless(eval(j,units)) 
        else (i,repr(brian2.get_dimensions(eval(j,units))))
        for i,j in dict(modeldict[k]).items()])

  deq= [[j,]+k.split(":") if k.count(":") else [j,k,varunits[j]] 
      for j,k in deq.items()]

#  deq= [(i,":".join( (str(sympy.S(j).subs(foos).subs(defs).subs(foos).subs(defs).subs(pars)),k))) for i,j,k in deq]
  
  deq= [(i,":".join( (str(rsubs(j,defs,foos,pars,r=r)),k) )) for i,j,k in deq]

  vode= sympy.S(sympy.solve(modeldict["kirschoff"],"dv/dt")[0])
  vode= rsubs(vode,defs,foos,pars,r=r)

  brianeq= brian2.Equations(
      "dv/dt={}".format(str(vode.subs(foos).subs(defs).subs(foos).subs(defs).subs(foos).subs(pars))+":volt"))
  for i,j in deq:
    brianeq+=brian2.Equations("d{}/dt={}".format(i,j))
  for i,j in dict(bifpar).items():
    j= eval(j,units)
    if brian2.is_dimensionless(j):
      brianeq+=brian2.Equations("{}:1".format(i,brian2.get_dimensions(j)))
    else:
      brianeq+=brian2.Equations("{}:{}".format(i,repr(brian2.get_dimensions(j))))
  return brianeq


