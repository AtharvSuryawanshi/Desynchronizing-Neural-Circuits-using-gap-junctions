import brian2
import sympy

from utils.brianutils import units


def rsubs(eq, *args, **kwargs):
    """
    Recursive substitutions in sympy. Applies the dictionaries holding
    the substitutions r-times.

    INPUT
      eq     : equation (sympy or str)
      args   : several dictionaries
      r      : is the recursion depth (default: r=1)
    """
    import sympy
    if "r" in kwargs:
        r = kwargs["r"]
    else:
        r = 1
    for j in range(r):
        for k in args:
            dic = [(str(i), str(j)) for i, j in dict(k).items()]
            eq = sympy.S(str(eq)).subs(dic)

    return eq


def load_model(model_dict, bifpar=[], substitution_depth=4, substitute_parameters=True):
    """ Loads dictionary holding model definitions into a
        brian2.Equations object.

        INPUT
          model_dict         : dictionary with model secifications. It must have
                               the following keys: ["ode", "definitions",
                               "functions", "parameters", "init_states"]
          bifpar             : list (or dict) with bifurcation parameters
          substitution_depth : recursion depth for substitution
        OUTPUT
          ode/sde            : brian2.Equations object
    """

    model_dict = dict(model_dict)
    mandatory_key_list = ["ode", "definitions", "functions", "parameters", "init_states"]
    optional_key_list = ["bibkey"]  # not used

    for key in mandatory_key_list:
        assert key in model_dict, 'model_dict is missing key "{}"'.format(key)

    parameter_dict = dict(model_dict["parameters"])

    brian_ode = brian2.Equations("")
    if bifpar:
        for key in bifpar:
            par_value = evaluate_string_as_brian2_expression(parameter_dict.pop(key))
            par_unit = get_parameter_unit(par_value)
            brian_ode += brian2.Equations("{0}:{1}".format(key, par_unit))

    if not substitute_parameters:
        parameters = list(parameter_dict.keys())
        for parameter in parameters:
            value_string = parameter_dict.pop(parameter)
            value = evaluate_string_as_brian2_expression(value_string)
            unit = get_parameter_unit(value)
            brian_ode += brian2.Equations("{name} = {value}:{unit}".format(**{"name": parameter,
                                                                                   "value": value_string,
                                                                                   "unit": unit}))

    statevar_list = list(model_dict["init_states"].keys())
    time_derivative_list = ["d{}/dt".format(k) for k in statevar_list]
    state_units = dict([[j, "1"] if brian2.is_dimensionless(eval(k, units))
                        else [j, repr(brian2.get_dimensions(eval(k, units)))]
                        for j, k in model_dict["init_states"].items()])

    for ode in model_dict["ode"]:
        odestr = "({})-({})".format(*ode.split("="))
        odestr = odestr.replace(" ", "")
        time_derivative = [k for k in time_derivative_list if k in odestr][0]  # bad style
        state_variable = [k for k in statevar_list if k in time_derivative][0]
        ode_rhs = sympy.S(sympy.solve(odestr, time_derivative)[0])  # [0] is bad style
        ode_rhs = rsubs(ode_rhs,
                        model_dict["definitions"],
                        model_dict["functions"],
                        parameter_dict,
                        r=substitution_depth)

        brian_ode += brian2.Equations("{0} = {1} : {2}".format(
            time_derivative,
            ode_rhs,
            state_units[state_variable])
        )

    return brian_ode


def evaluate_string_as_brian2_expression(value_string):
    return eval(value_string, units, vars(brian2))


def get_parameter_unit(par_value):
    if brian2.is_dimensionless(par_value):
        par_unit = repr(1)
    else:
        par_unit = repr(brian2.get_dimensions(par_value))
    return par_unit


def equations2dict(ode):
    """ equations2dict(ode)
        returns a dict with the model equation that can be saved
        with json and is compatible with load_model.

        INPUT
          ode : (brian2.Equations)
        OUTPUT
          model-dictionary (dict)
    """
    return {
        "ode": ["d{0}/dt = {1}".format(j, k) for j, k in ode.eq_expressions],
        "init_states": dict([(j, repr(ode.dimensions[j])) for j in ode.eq_names]),
        "functions": {},
        "parameters": dict([(j, repr(ode.dimensions[j])) for j in ode.parameter_names]),
        "definitions": {}
    }


def funcsubs(expr, dic):
    from sympy import parse_expr
    if type(expr)==str:
        expr = parse_expr(expr)
    dic= {j:parse_expr(dic[j]) if type(dic[j])==str else dic[j] for j in dic}
    for j, k in dic.items():
        if type(j)==str:
            j=parse_expr(j)
        expr = expr.replace(j, k)
    return expr


def loadmod(model_dict, bifpar=[], substitution_depth=4):
    """ Loads dictionary holding model definitions into a
        brian2.Equations object. This is the *newer* version
        using the format with diff(v(t),t) on the rhs. The purpose is to allow
        for symbolic manipulation of the ODEs.
        INPUT
          model_dict         : dictionary with model secifications. It must have
                               the following keys: ["ode", "definitions",
                               "functions", "parameters", "init_states"]
          bifpar             : list (or dict) with bifurcation parameters
          substitution_depth : recursion depth for substitution
        OUTPUT
          ode/sde            : brian2.Equations object
    """
    from sympy.parsing.sympy_parser import parse_expr
    from sympy import Function, Symbol, Eq, Wild, solve, preorder_traversal
    model_dict = dict(model_dict)
    mandatory_key_list = ["ode", "definitions", "functions", "parameters", "init_states"]
    optional_key_list = ["bibkey"]  # not used
    for key in mandatory_key_list:
        assert key in model_dict, 'model_dict is missing key "{}"'.format(key)
    foodict = {}
    for fstr, expr in model_dict['functions'].items():
        f_of_x = parse_expr(fstr)
        foo = Function(repr(f_of_x.func))
        arg = [Wild(repr(k)) for k in f_of_x.args]
        foodict[foo(*arg)] = parse_expr(expr, {repr(k): k for k in arg})
    parameter_dict = dict(model_dict["parameters"])
    brian_ode = brian2.Equations("")
    if bifpar:
        for key in bifpar:
            value_string = parameter_dict.pop(key)
            if isinstance(value_string, str):
                par_value = eval(value_string, units, vars(brian2))
            else:
              par_value = value_string
            if brian2.is_dimensionless(par_value):
                par_unit = repr(1)
            else:
                par_unit = repr(brian2.get_dimensions(par_value))
            brian_ode += brian2.Equations("{0}:{1}".format(key, par_unit))
    diffeqs = [Eq(*[parse_expr(k) for k in r.split("=")]) for r in model_dict['ode']]
    diffeqs = [k.subs(model_dict["definitions"]) for k in diffeqs]
    differential = [k for j in diffeqs for k in preorder_traversal(j) if k.is_Derivative]
    rhs = [solve(j, k)[0] for j, k in zip(diffeqs, differential)]
    var_t = [k.args[0] for k in differential]
    var = [Symbol(repr(k.func)) for k in var_t]
    var2var_t = {j: k for j, k in zip(var, var_t)}
    var_t2var = {j: k for j, k in zip(var_t, var)}
    rhs_ss = [funcsubs(k, var_t2var) for k in rhs]
    state_units = dict([[j, "1"] if brian2.is_dimensionless(eval(k, units))
                        else [j, repr(brian2.get_dimensions(eval(k, units)))]
                        for j, k in model_dict["init_states"].items()])
    for var_k, rhs_k in zip(var, rhs_ss):
        rhs_k = rhs_k.subs(model_dict["definitions"])
        rhs_k = funcsubs(rhs_k, foodict)
        rhs_k = rhs_k.subs(parameter_dict)
        brian_ode += brian2.Equations("d{0}/dt = {1} : {2}".format(
            var_k, rhs_k, state_units[repr(var_k)])
        )
    return brian_ode
