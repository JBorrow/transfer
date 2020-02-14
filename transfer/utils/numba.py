"""
Numba utilities for :mod:`transfer`.

These are extra functions that are not provided as part of the
numba 'standard library'.
"""

import numba


@numba.njit
def create_numba_hashtable(left, right):
    """
    Creates a numba-dict hashtable from left: right, such that
    hashtable[left] = right.

    This is required as the regular python ``dict`` is _not_
    the same as the ``numba`` ``Dict``, which is accelerated.

    Parameters
    ----------

    left: hashable
        Hashtable keys

    right
        Values in hashtable


    Returns
    -------

    hashtable: numba.typed.Dict
        Hashtable from [left]: right
    """

    output = dict()

    for i in range(left.size):
        output[left[i]] = right[i]

    return output
