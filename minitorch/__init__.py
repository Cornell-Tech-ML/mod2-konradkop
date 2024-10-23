"""Module provides a comprehensive suite of tools for tensor operations,
automatic differentiation, optimization, and related functionalities.

Modules included:
- tensor: Core tensor operations and data structures.
- tensor_data: Utilities for tensor data handling.
- tensor_ops: Fundamental tensor operations.
- tensor_functions: Various tensor-related mathematical functions.
- datasets: Datasets for testing and training.
- optim: Optimization algorithms for training models.
- autodiff: Automatic differentiation utilities.
- scalar: Operations for scalar values.
- scalar_functions: Functions for scalar computations.
- testing: Testing utilities for validating tensor operations and functions.
- module: Neural network modules and layers.

This module aims to provide efficient and user-friendly interfaces for
machine learning and numerical computing tasks.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
