"""Internal location of the b12x compatibility implementation.

Consumers import ``b12x``. Internal imports share its module identity.
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("b12x")
