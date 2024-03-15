# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

from functools import wraps


def wrap_numba_func():
    """
    wrapper for numba functions. The decorated function will be able to accept any inputs having the __array__ protocol.
    """


def auto_align(fn):
    """
    Automatically align the inputs of the function.

    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        pass

    return wrapper
