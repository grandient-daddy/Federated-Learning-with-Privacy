from collections import Counter
from itertools import accumulate
from bisect import bisect_left
from math import log, log1p

import numpy as np
import pandas as pd

from lib import index, searchranked
ln2 = log(2)


def get_unknown_items(udx, unseen_interactions):  
    """
    This function is necessary for compatibility.
    """
    return unseen_interactions[udx]


def get_known_items(udx, seen_interactions):
    """
    Gets array of items that user(udx) interacted with.
    
    Parameters
    ----------
    udx : int
        Index of a particular user in a system
    
    seen_interactions : scipy.sparse.csr_matrix
        User/Item binary preference matrix.

    Returns
    -------
    indices : numpy.ndarray
        Array of known user items
    """
    indptr = seen_interactions.indptr
    indices = seen_interactions.indices    
    return indices[indptr[udx]:indptr[udx + 1]]


def find_target_rank(target_item, generate_scores, uid, sid, sess_items, item_pool, topk):
    """
    Finds rank of a target item compared to items from item_pool.
    
    Parameters
    ----------
    target_item : int
        Index of an item.

    generate_scores : function
        Function that generates scores s.t.:
        generate_scores(uid, sid, sess_items, items_pool) -> array of scores.

    uid : int
        Index of a particular user in a system.

    sid : int
        Index of a particular session for a particular user.

    sess_items : list
        Item indices list representing a session.

    item_pool : numpy.ndarray
        Array of item indices.

    topk : int
        The number of elements to select for metric.

    Returns
    -------
    item_rank : int
        Rank value of an item.
    """
    predict_for_items = item_pool
    target_pos = index(item_pool, target_item)
    if target_pos is None: # i.e., app is not installed
        # two possible cases:
        # - target item is known in general, but not installed on this device
        # - completely new app, not seen in training, i.e. no index in global model
        # both cases must be gracefully handled by a model
        target_pos = len(item_pool) // 2 # avoid boundary positions - helps catching bugs
        predict_for_items = np.r_[item_pool[:target_pos], target_item, item_pool[target_pos:]]
    scores = generate_scores(uid, sid, sess_items, predict_for_items)
    item_rank = searchranked(scores, target_pos, topk)
    return item_rank


def metric_increments(ranks, numk, topk_bins):
    """
    Groups and accumulates metrics in bins corresponding to top-k intervals,
    e.g., for topk=[1,3,5] it will accumulate into intervals [1], [2-3], [4-5].
    Then, in order to obtain "<=k" intervals, one has to compute a running sum.
    Relies on ranks containing values not larger than max topk. 

    Parameters
    ----------
    ranks : list
        List of items ranks.

    numk : int
        The number of different 'k' for metric@k.

    topk_bins : list
        List of top-k bins.

    Returns
    -------
    hits : list
        List of hits.

    reci : list
        List of reciprocal ranks.

    reci_log1p : list
        List of reciprocal ranks with applied logarithms.
    """
    hits = [0] * numk
    reci = [0] * numk
    reci_ln1p = [0] * numk
    counter = Counter(ranks)
    counter.pop(0, None) # remove invalid rank if present
    for rank, freq in counter.items(): # groups of ranks and their frequencies
        k_bin = topk_bins[rank] # a bin the group belongs to based on topk interval
        hits[k_bin] += freq
        reci[k_bin] += freq / rank
        reci_ln1p[k_bin] += freq / log1p(rank) # log1p = ln(1+x)
    reci_log1p = [ln2*x for x in reci_ln1p] # log2(1+x) = log(1+x) / log(2)
    return hits, reci, reci_log1p


def incremental_metrics(topk_list):
    """
    Initializes metric scores dictionary and gets scores updater.

    Parameters
    ----------
    topk_list : list
        Sorted list of top-k values.

    Returns
    -------
    scorer : function
        Function to recalculate metric scores.

    metric_dict : dict
        Initialized dictionary of metric scores.
        
    Note: map metric is not computed, as map equals mrr for 1 holdout item.
    example: https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank
    """
    numk = len(topk_list)
    metrics = [hr, mrr, ndcg] = [[0]*numk, [0]*numk, [0]*numk]
    topk_bins = [bisect_left(topk_list, x) for x in range(topk_list[-1]+1)]

    def scorer(ranks):
        cnt = len(ranks)
        scorer.counts += cnt
        increments = metric_increments(ranks, numk, topk_bins)
        # accumulate average values incrementally
        hr[:], mrr[:], ndcg[:] = (
            [val + (inc-cnt*val) / scorer.counts for val, inc in zip(metric, increment)]
            for metric, increment in zip(metrics, map(accumulate, increments))
        )
    scorer.counts = 0
    return scorer, {'hr': hr, 'mrr': mrr, 'ndcg': ndcg}


def collect_metrics(generate_scores, test_sessions, interactions, topk_list, mode):
    """
    Computes metrics for particular test sessions data and aggregates them into dictionaries
    of ranks(raw data) and metric scores for any user.
    
    Parameters
    ----------
    generate_scores : function
        Function that generates scores s.t.:
        generate_scores(uid, sid, sess_items, items_pool) -> array of scores.
    
    test_sessions : pandas.Series
        Data structure representing users' sessions s.t.
        test_sessions[userid] = [[item1, item3, ...], [item1, ..., item6], [item3, ...], ...]
     
    interactions : scipy.sparse.csr_matrix
        User/Item binary preference matrix.

    topk_list : list
        Sorted list of top-k values.

    mode: str
        "known_items" to get recommendation for seen items or "unknown_items" to get negative samples.

    Returns
    -------
    user_metrics : dict
        Dictionary of the user specific metrics.

    user_stats : dict
        Dictionary of the user specific ranks over sessions.
    """
    
    # choose evaluation mode:
    if mode == 'known_items':
        get_item_pool = get_known_items
    elif mode == 'unknown_items':
        get_item_pool = get_unknown_items
    else:
        assert False, "Bad mode"
    
    # get the dictionary of item positions:  
    user_metrics = {}
    user_stats = {}
    max_topk = topk_list[-1]

    sid_increment = 0
    for uid, sessions in test_sessions.items():
        user_stats[uid] = stats = []
        metrics_updater, metrics = incremental_metrics(topk_list)
        item_pool = get_item_pool(uid, interactions) # i.e., all installed apps on user device/negative examples
        for sid, session in enumerate(sessions):
            sess_ranks = []
            sess_items = []
            for i in range(len(session)-1):
                sess_items.append(session[i])
                target_rank = find_target_rank(
                    session[i+1], generate_scores,
                    uid, sid_increment+sid, sess_items,
                    item_pool, max_topk
                )
                sess_ranks.append(target_rank)
            metrics_updater(sess_ranks)
            stats.append(sess_ranks)
        sid_increment += len(sessions)
        user_metrics[uid] = metrics
    return user_metrics, user_stats


def evaluate(generate_scores, test_sessions, interactions, topk=(1, 3, 5), mode="known_items"):
    """
    Evaluate the model using session-based methodology.
    
    Parameters
    ----------
    generate_scores : function
        Function that generates scores s.t.:
        generate_scores(uid, sid, sess_items, items_pool) -> array of scores.
    
    test_sessions : pandas.Series
        Data structure representing users' sessions s.t.
        test_sessions[userid] = [[item1, item3, ...], [item1, ..., item6], [item3, ...], ...]
     
    interactions : scipy.sparse.csr_matrix
        User/Item binary preference matrix.

    topk : list
        List of top-k values to evaluate against.

    mode: str
        "known_items" to get recommendation for seen items or "unknown_items" to get negative samples.
        Note: you need "known_items".

    Returns
    -------
    metrics_df : pandas.Dataframe
        Dataframe of the following structure: metrics_df.loc[userid][metric][k]

    user_stats : dict
        Dictionary of the user specific ranks over sessions.
    """
    topk = sorted(topk)
    user_metrics, user_stats = collect_metrics(generate_scores, test_sessions, interactions, topk, mode)
    metrics_df = pd.concat(
        {user: pd.DataFrame(metrics, index=topk) for user, metrics in user_metrics.items()},
        names=['userid', 'topk']
    ).rename_axis(columns='metrics')
    return metrics_df, user_stats


## For experiments ##
def append_result(stat_dict, metrics, factor_norms):
    """
    Updates 'stat_dict' with new metric results from 'metrics'.
    """
    for metric in metrics.keys():
        for topk in metrics[metric].keys():
            stat_dict[metric][topk].append(metrics[metric][topk])
    
    for factor, norm in factor_norms.items():
        stat_dict["norms"][factor].append(norm)

def get_stat_dict(metric_names, topk, factor_names):
    """
    Initializes 'stat_dict' structure with 'metric_names' and 'topk' values.
    """
    stat = {metric: {k: [] for k in topk} for metric in metric_names}
    stat["norms"] = {name: [] for name in factor_names}
    return stat
 
def nm_zero(fact):
    """
    Computes norm of a "fact" if it is an numpy.ndarray
    else returns -1.
    """
    if isinstance(fact, np.ndarray):
        return np.linalg.norm(fact)
    return -1

def evaluation_callback(
    get_scores_generator,
    test_sessions,
    interactions,
    factor_names,
    scorer_mode,
    eval_mode="known_items",
    rmse_loss=None,
    rmse_params=None, 
):
    """
    Initializes evaluation callback.
    
    Parameters
    ----------
    get_scores_generator : function
        get_scores_generator function from seqmf_pp.py or mf.py
    
    test_sessions : pandas.Series
        Data structure representing users' sessions s.t.
        test_sessions[userid] = [[item1, item3, ...], [item1, ..., item6], [item3, ...], ...]
     
    interactions : scipy.sparse.csr_matrix
        User/Item binary preference matrix.

    factor_names : list
        List of factors names, e.g. ["P", "P1", "Q", "Q1"]

    scorer_mode : str
        Mode for get_scores_generator function, depending on a model.

    eval_mode : str, optional
        Default is "known_items", use it for session-based evaluation.

    rmse_loss : function
        Loss function s.t.
        rmse_loss(local_factors, global_factors, **rmse_params) -> float.

    rmse_params : dict
        Dictionary of parameters for rmse_loss.

    Returns
    -------
    callback : function
        Function to be called on every epoch/iteration
        to check quality metrics, norms and loss s.t.
        callback(local_factors, global_factors) -> None
    """
   
    def callback(local_factors, global_factors):
        scorer = get_scores_generator(local_factors, global_factors, scorer_mode)
        metrics_df, user_stats = evaluate(scorer, test_sessions, interactions, mode=eval_mode)
        factor_norms = {fname: nm_zero(fact) for fname, fact in zip(factor_names, local_factors + global_factors)}
        results = (
            metrics_df
            .reset_index()
            .groupby(["topk"])
            .mean()[["hr", "mrr", "ndcg"]].to_dict()
        )
        append_result(callback.stat, results, factor_norms)
        if rmse_loss:
            callback.rmse.append(rmse_loss(local_factors, global_factors, **rmse_params))
    
    callback.stat = get_stat_dict(
        metric_names=["hr", "mrr", "ndcg"],
        topk=[1, 3, 5],
        factor_names=factor_names
    )
    callback.rmse = []
    return callback
## For experiments ##