import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


session_break_delta = '30min'
invalid_rank = 0
terminal_item = -1


def drop_consequtive_repeats(
    df,
    window=None,
    user_col='userid',
    item_col='appid',
    time_col='timestamp',
):
    """
    Drops all consecutive repeated events except the first one.
    If window is defined, repeated events outside the time window
    will be preserved, not dropped.
    
    Note: function assumes data is sorted by users and timestamps.
    """
    reps = (df[item_col].diff() == 0) & (df[user_col].diff() == 0)
    if window is not None:
        window = pd.Timedelta(window)
        reps = reps & (df[time_col].diff() <= window)
    return df.drop(reps.index[reps])


def assign_session_id(
    df,
    session_break_delta,
    user_col='userid',
    time_col='timestamp',
):
    """
    The function assigns session column to dataframe based on session_break_delta.

    Note: function assumes data is sorted by users and timestamps.
    """
    return df.assign(
        sessid=lambda x: ( # split into sessions based on time interval
                (
                    (x[time_col].diff() > pd.Timedelta(session_break_delta)) 
                    & (x[user_col].diff() == 0)
                )
            ).groupby(x[user_col]).cumsum()
    )


def select_previous_items(data, itemid, level):
    """
    Find previous items data using different levels e.g. user or session levels.
    """
    prev_items_data = (
        data[itemid]
        .shift(1, fill_value=terminal_item)
    )
    prev_items_data.loc[data[level].diff()!=0] = terminal_item # exclude inter-level connections
    return prev_items_data   


def generate_transition_matrices(
    data,
    pow_bool,
    shape=None,
    level=None,
    userid='userid',
    itemid='appid',
    exclude_item=None
):
    """
    Generate dictionary of transition matrices(S in the paper).
    
    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe that has at least 2 columns related to object and subject, e.g. item and user.
        Level column should be taken into account as well.
    
    pow_bool : bool
        Raise normalization constant into the power of 0.5 or not.
    
    shape : tuple, optional
        Shape of transition matrix.
        Default is None

    level : str, optional
        What level to use for session splitting and matrices generation e.g. 
        'userid' - global level(one session), 'sessid' - local sessions.
        Default is None. It supposes level=userid. 
    
    userid : str, optional
        Name of user column in the 'data' Dataframe
        Default is 'userid'

    itemid : str, optional
        Name of item column in the 'data' Dataframe
        Default is 'appid'

    exclude_item : object
        Default is None. Use this.

    Returns
    -------
    transition_data : dict
        Dictionary of user related transition matrices.
    """   
    level = level or userid
    prev_items = select_previous_items(data, itemid, level)
    
    transition_df = pd.DataFrame(
        {'_to': data[itemid], '_from': prev_items},
        index=data.index, copy=False
    )
    items_mask = slice(None)
    if exclude_item is not None:
        items_mask = prev_items != exclude_item # e.g., skip breaks 
    if shape is None:
        shape = tuple(transition_df.loc[items_mask].max()[['_to', '_from']]+1)
    
    transition_data = {}
    grouper = data[level] if level == userid else [data[userid], data[level]]
    for group_id, item_data in transition_df.loc[items_mask].groupby(grouper):
        transitions = (
            item_data
            .groupby('_to')['_from'] # form single-item groups (i.e. only consider most recent items)
            .value_counts(normalize=True) # group frequencies
        )
        if pow_bool:
            transitions = transitions.pow(0.5)
        rows = transitions.index.get_level_values(0)
        cols = transitions.index.get_level_values(1)
        vals = transitions.values
        transition_data[group_id] = csr_matrix((vals, (rows, cols)), shape=shape)
    return transition_data


def list_session_lists(data, userid='userid', itemid='appid', sessid='sessid', min_length=0):
    """
    Create sessions representation using Pandas.Series that has the following structure:
    result[userid] = [[item1, item3, ...], [item1, ..., item6], [item3, ...], ...]

    Note: function assumes data is sorted by users and timestamps.
    """
    return (
        data
        .groupby([userid, sessid])[itemid]
        .apply(list) # represent sessions as lists of items
        .loc[lambda x: x.apply(len)>=min_length] # filter sessions with <`min_length` items
        .groupby(level=userid)
        .apply(list) # represent users as lists of sessions
    )


def get_user_pref(data, item_col):
    """
    Get known items pool from data.
    Can be used to get user preferences.
    """
    items = data[item_col]
    return items.unique()