"""This is a collection of function used in combination with brian2."""
from sys import version_info
if version_info[0] < 3:
  from future import *

from .brianutils import *
from .load_model import *

import sympy, brian2, warnings, json, numpy

units= dict(
      list(vars(brian2.units).items())
    + list(vars(brian2.units.allunits).items())
    + list(vars(brian2.units.fundamentalunits).items())
    + list(vars(brian2).items())
    )
