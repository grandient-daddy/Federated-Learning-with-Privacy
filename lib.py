from itertools import takewhile
import numpy as np
from numba import njit

@njit
def index(array, item):
    """
    Find index of an item in an array.
    
    Parameters
    ----------
    array : iterable
        Any iterable object, e.g. numpy.ndarray, list, tuple.
    
    item : object
        Item that we seek in an array.

    Returns
    -------
    idx : int
        Index of an item.
    """
    for idx, val in enumerate(array):
        if val == item:
            return idx

def searchranked(array, target_pos, topk):
    """
    Finds a rank of target (located at 'target_pos') in 'array'
    Pessimizes rank if there're other elements in 'array' equal to target.

    Parameters
    ----------
    array : iterable
        Any iterable object, e.g. numpy.ndarray, list, tuple.
    
    target_pos : int
        Position of a target.

    topk : int
        The number of elements to select

    Returns
    -------
    pos : int
        Rank value if target's among topk elements, otherwise returns 0.
    
    Note: rank values start from 1, meaning "the first among topk elements".
    """
    target = array[target_pos]
    vals = takewhile(# select no more than topk elemenets with values >= target value
        lambda x: x[0]<=topk, # for identical scores, `<=` pushes target out of top-k list
        enumerate(a for a in array if a >= target) # allows identical scores to push target out
    )
    pos = len(list(vals)) # always >= 1, because `target` itself is in `array`
    return pos if pos <= topk else 0


def topsort(a, topk):
    """
    Find top-k indices of array 'a' elements.

    Parameters
    ----------
    a : numpy.ndarray
        Array with numbers.

    topk : int
        The number of elements to select

    Returns
    -------
    parted : numpy.ndarray
        Indices of the largest k elements in an array
    """
    parted = np.argpartition(a, -topk)[-topk:]
    return parted[np.argsort(-a[parted])]
