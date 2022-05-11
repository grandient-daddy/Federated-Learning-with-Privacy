import logging
import numpy as np

import dataprep

from numba import njit
from scipy.sparse import coo_matrix

from dataprep import generate_transition_matrices, get_user_pref


## For exceptions ##
class MethodError(Exception):
    def __init__(self, params):
        self.params = params
## For exceptions ##


## Local factors update ##
def contextual_preferences(S, Q):
    """
    The function produces sequential scores (diag(SQQ^T))
    and intermediate data.
    """
    SQ = S @ Q
    return np.einsum('ij,ij->i', SQ, Q, optimize=False), SQ

def build_linear_equation_P(C, cu, S_dict, Q, R, u):
    """
    The function produces linear system matrix and right hand side to
    find local factor for user 'u' using ALS.
    """
    indptr = C.indptr
    inds = C.indices[indptr[u]:indptr[u + 1]]
    coef = C.data[indptr[u]:indptr[u + 1]]

    # calculate A
    Qnnz = Q[inds]
    Qcu = Q * cu[u]
    QtQ_R = R + Q.T.dot(Qcu)
    A_sys = QtQ_R + Qnnz.T.dot(coef[:, np.newaxis] * Qnnz)

    # seq
    S = S_dict.get(u)
    if S is None:
        seq = None
    else:
        seq, _ = contextual_preferences(S, Q)

    # calculate b
    QtC = Qnnz.T * coef
    b_sys = QtC.sum(axis=1) + Qcu[inds].T.sum(axis=1)
    if seq is not None:
        b_sys -= QtC.dot(seq[inds])
        b_sys -= Qcu.T.dot(seq)
    return A_sys, b_sys

def least_squares_P(C, cu, S_dict, Q, R, u):
    """
    The function produces local factor for user 'u' using ALS.
    """
    A_sys, b_sys = build_linear_equation_P(C, cu, S_dict, Q, R, u)
    try:
        res = np.linalg.solve(A_sys, b_sys)
    except np.linalg.LinAlgError as err:
        error_dict = {
            "error_type": err,
            "A norm": np.linalg.norm(A_sys),
            "b_norm": np.linalg.norm(b_sys),
        }
        raise MethodError(error_dict)
    return res

def update_P(P, C, cu, S_dict, Q, regI):
    """
    Inplace ALS update of local factors(P)
    
    Parameters
    ----------
    P : numpy.ndarray
        Local parameters for inplace update
    
    C : scipy.sparse.csr_matrix
        Confidence weight matrix s.t. 
        C_{ui} = f_{ui}^{gamma} / (sum_j(f_{uj}^{gamma}) + \alpha*n_items)
    
    cu : numpy.ndarray
        Laplacian smoothing part of confidence weight s.t.
        cu_u = \alpha / (sum_j(f_{uj}^{gamma}) + \alpha*n_items)

    S_dict : dict
        Dictionary of “design matrices" encoding relative frequency weights
        corresponding to transition to item i from some previous item in user history,
        e.g. S_dict[user_id] = csr_matrix(shape=(n_items,)*2)

    Q : numpy.ndarray
        Global parameters matrix

    regI : numpy.ndarray
        Regularization diagonal matrix.
    """
    n_users, _ = C.shape
    for u in range(n_users):
        P[u] = least_squares_P(C, cu, S_dict, Q, regI, u)
## Local factors update ##


## Global factors update ##
def grad_update(Q, C, cu, S_dict, P, u):
    """
    The function produces gradient related to user 'u'.
    """
    indptr = C.indptr
    inds = C.indices[indptr[u]:indptr[u + 1]]
    coef = C.data[indptr[u]:indptr[u + 1]]

    S = S_dict.get(u)
    p_u = P[u]
    r_u = Q.dot(p_u)

    if S is not None:
        seq, SQ = contextual_preferences(S, Q)
        r_u += seq

    r_u[inds] -= 1
    r_u_const = r_u * cu[u]
    r_u[inds] *= coef
    Du = r_u + r_u_const
    grad = np.outer(Du, p_u)
    if S is not None:
        DuSQ = Du[:, np.newaxis] * SQ
        StDuQ = (S.T * Du).dot(Q)
        grad += DuSQ + StDuQ
    return grad

def get_grad_Q(Q, C, cu, S_dict, P, reg):
    """
    The function produces gradient related to the sum of all the users in a system
    """
    grad = reg * Q
    n_users, _ = C.shape
    for user in range(n_users):
        grad += grad_update(Q, C, cu, S_dict, P, user)
    return grad

@njit
def adam(grad, cum_grad, cum_sq_grad, beta1, beta2, smoothing):
    """
    The function produces adjusted gradient computed using Adam
    """
    cum_grad_update = beta1 * cum_grad + (1 - beta1) * grad
    cum_sq_grad_update = beta2 * cum_sq_grad + (1 - beta2) * (grad * grad)
    db1 = 1 - beta1
    db2 = 1 - beta2
    adjusted_grad = cum_grad_update/db1 / (smoothing + np.sqrt(cum_sq_grad_update/db2))
    return adjusted_grad, cum_grad_update, cum_sq_grad_update

def update_Q(Q, C, cu, S_dict, P, lr, reg, n_steps, adam_config):
    """
    Inplace Gradient Descent(Adam) update of global factors(Q)
    
    Parameters
    ----------
    Q : numpy.ndarray
        Global parameters for inplace update
    
    C : scipy.sparse.csr_matrix
        Confidence weight matrix s.t. 
        C_{ui} = f_{ui}^{gamma} / (sum_j(f_{uj}^{gamma}) + \alpha*n_items)
    
    cu : numpy.ndarray
        Laplacian smoothing part of confidence weight s.t.
        cu_u = \alpha / (sum_j(f_{uj}^{gamma}) + \alpha*n_items)

    S_dict : dict
        Dictionary of “design matrices" encoding relative frequency weights
        corresponding to transition to item i from some previous item in user history,
        e.g. S_dict[user_id] = csr_matrix(shape=(n_items,)*2)

    P : numpy.ndarray
        Local parameters matrix

    lr : float
        Learning rate for gradient based optimizer
    
    reg : float
        Tikhonov regularization parameter

    n_steps : int
        The number of Q full updates


    adam_config : dict
        Dictionary of adam optimizer parameters 
    """
    for _ in range(n_steps):
        grad = get_grad_Q(Q, C, cu, S_dict, P, reg)
        if adam_config:
            grad, adam_config['cum_grad'], adam_config['cum_sq_grad'] = adam(grad, **adam_config)
        Q -= lr * grad
## Global factors update ##


## Data preparation functions ##
def get_conf_mtx_lap_smooth(
    data,
    n_subject,
    n_object,
    subject_col_name,
    object_col_name,
    alpha,
    gamma,
):
    """ 
    Function to generate confidence weights matrix.
    
    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe that has at least 2 columns related to object and subject,
        e.g. item and user
    
    n_subject : int
        The number of unique subjects in the system

    n_object : int
        The number of unique objects in the system

    subject_col_name : str
        Name of subject column in the 'data' Dataframe

    object_col_name : str
        Name of object column in the 'data' Dataframe

    alpha : float
        Alpha parameter of laplacian smoothing

    gamma : float
        Allows to increase or decrease the influence of popular items

    Returns
    -------
    C : scipy.sparse.csr_matrix
        Confidence weight matrix s.t. 
        C_{ui} = f_{ui}^{gamma} / (sum_j(f_{uj}^{gamma}) + \alpha*n_items)

    cu : numpy.ndarray
        Laplacian smoothing part of confidence weight s.t.
        cu_u = \alpha / (sum_j(f_{uj}^{gamma}) + \alpha*n_items)

    Note: data is assumed to be already reindexed! 
    """
    shape = n_subject, n_object
    rows, cols = data[subject_col_name], data[object_col_name]
    freq_matrix = (
        coo_matrix(
            (
                np.ones_like(rows),
                (rows, cols),
            ),
            shape=shape,
        ).tocsr()
    )
    freq_matrix = freq_matrix._with_data(np.power(freq_matrix.data, gamma))
    denom = 1 / (freq_matrix.sum(axis=1) + n_object * alpha)
    freq_matrix = freq_matrix.multiply(denom).tocsr()
    return freq_matrix, (alpha * denom).A1

def get_users_sui(
    data,
    n_items,
    user_col,
    item_col,
    pow_bool,
    level=None
):
    """
    Function to generate dictionary of “design matrices" encoding relative frequency weights
    corresponding to transition to item i from some previous item in user history.
    
    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe that has at least 2 columns related to object and subject,
        e.g. item and user

    n_items : int
        The number of unique objects in the system

    user_col : str
        Name of user column in the 'data' Dataframe

    item_col : str
        Name of item column in the 'data' Dataframe

    pow_bool : bool
        Divide by the squared root of normalization constant or not

    level : str, optional
        Level which is used to create design matrix, 
        e.g. 'userid' or 'sessid'
        Default is None (level=user_col)

    Returns
    -------
    S_dict: dict
        Dictionary of “design matrices".

    Note: data is assumed to be already reindexed!
    """
    kwargs = dict(
        shape = (n_items,)*2,
        userid = user_col,
        itemid = item_col,
        exclude_item = dataprep.terminal_item
    )
    return generate_transition_matrices(data, pow_bool, level=level, **kwargs)
## Data preparation functions ##


## Main model ##
def seq_mf_pp(
    data,
    n_users,
    n_items,
    user_col,
    item_col,
    pow_bool,
    n_factors,
    n_epochs,
    regularization,
    lap_smooth,
    gamma,
    lr,
    n_steps,
    seed=0,
    evaluation_callback=None,
    iterator=range,
    optimizer_mode='sgd',
):

    """
    A Sequence-Aware Recommender System using hybrid training strategy: ALS for local factors P,
        GD for global factors Q.

    Parameters
    ----------
    data : pandas.Dataframe
        Interactions Dataframe that has at least 2 columns related to object and subject, e.g. item and user

    n_users : int
        The number of unique subjects(users) in the system

    n_items : int
        The number of unique objects(items) in the system

    user_col : str
        Name of subject column in the 'data' Dataframe

    item_col : str
        Name of object column in the 'data' Dataframe

    pow_bool : bool
        Divide by the squared root of normalization constant or not

    n_factors : int
        The dimensionality of a latent space

    n_epochs : int
        The number of full updating cycles

    regularization : float
        The Tikhonov regularization constant

    lap_smooth : float
        Alpha parameter of laplacian smoothing

    gamma: float
        Allows to increase or decrease the influence of popular items

    lr : float
        Learning rate of a GD updates part

    n_steps : int
        How many times to update global factors per epoch

    seed : int
        The random state for seeding the initial item and user factors.

    evaluation_callback : function, optional
        The callback with the following design: callback(local_factors, global_factors)
        Default is None. 

    iterator : iterable, optional
        Iterator used for tracking epochs.
        Default is 'range'.

    optimizer_mode: str, optional
        Optimizator to use: 'sgd' or 'adam'

    Returns
    -------
        local_factors : tuple
        Factors are: (P, None)

        global_factors : tuple
        Factors are: (Q, None)
    """

    Cui, cu = get_conf_mtx_lap_smooth(
        data,
        n_users,
        n_items,
        user_col,
        item_col,
        lap_smooth,
        gamma,
    )

    S_dict = get_users_sui(
        data,
        n_items,
        user_col,
        item_col,
        pow_bool=pow_bool,
        level=user_col,
    )


    assert Cui.shape == (n_users, n_items)

    #Initialize factors:
    random_state = np.random.RandomState(seed)
    P = random_state.normal(0, 0.001, size=(n_users, n_factors))
    Q = random_state.normal(0, 0.001, size=(n_items, n_factors))

    regI = np.zeros((n_factors,)*2)
    regI[np.diag_indices_from(regI)] += regularization

    #Initialize adam params if needed:
    if optimizer_mode == 'adam':
        adam_config = {
            'cum_grad': np.zeros_like(Q),
            'cum_sq_grad': np.zeros_like(Q),
            'beta1': 0.9,
            'beta2': 0.999,
            'smoothing': 1e-12,
        }
    elif optimizer_mode == 'sgd':
        adam_config = None
    else:
        assert False, 'Bad optimizer mode!'


    #Start trainig procedure:
    if evaluation_callback:
        evaluation_callback((P, None), (Q, None))

    for ep in iterator(n_epochs):

        # update P:
        try:
            update_P(P, Cui, cu, S_dict, Q, regI)
        except MethodError as err:
            print("Error:", err.params)
            return (np.zeros_like(P), None), (np.zeros_like(Q), None)

        if evaluation_callback:
            evaluation_callback((P, None), (Q, None))

        # update Q:
        update_Q(Q, Cui, cu, S_dict, P, lr, regularization, n_steps, adam_config)

        if evaluation_callback:
            evaluation_callback((P, None), (Q, None))

    return (P, None), (Q, None)

def get_scores_generator(local_factors, global_factors, mode="sum", pow_bool=False):
    """
    Create scores generator for the provided local and global factors.

    Parameters
    ----------
    local_factors : tuple
        Tuple of local factors s.t. (P, W) where P is used for prediction

    global_factors : tuple
        Tuple of global factors s.t. (Q, Qb) where Q is used for prediction

    mode : str, optional
        The recommendation mode has the following options:
        - 'last': takes into account the last item in a provided session list
        - 'sum': takes into account the sum of all the items in a provided session list
        - 'av': takes into account the average of all the items in a provided session list
        - 'seq_av': takes into account the average of all the items without long term preferences
        - 'seq_sum': takes into account the sum of all the items without long term preferences
        - 'no_seq': Does not take into account sequential information, only long term preferences
        Default is 'sum'.

    Returns
    -------
    generate_scores : function
        Function to generate scores for a particular user, session, items.
    """
    P, _ = local_factors
    Q, _ = global_factors

    if mode == "last":
        def generate_scores(uid, sid, sess_items, item_pool):
            scores = Q[item_pool] @ (P[uid] + Q[sess_items[-1]])
            return scores
        return generate_scores

    if mode == "sum":
        def generate_scores(uid, sid, sess_items, item_pool):
            scores = Q[item_pool] @ (P[uid] + Q[sess_items].sum(axis=0))
            return scores
        return generate_scores

    if mode == "av":
        def generate_scores(uid, sid, sess_items, item_pool):
            if pow_bool:
                scores = Q[item_pool] @ (P[uid] + Q[sess_items].sum(axis=0)/np.sqrt(len(sess_items)))
            else:
                scores = Q[item_pool] @ (P[uid] + Q[sess_items].mean(axis=0))
            return scores
        return generate_scores

    if mode == "seq_av":
        def generate_scores(uid, sid, sess_items, item_pool):
            if pow_bool:
                scores = Q[item_pool] @ (Q[sess_items].sum(axis=0)/np.sqrt(len(sess_items)))
            else:
                scores = Q[item_pool] @ Q[sess_items].mean(axis=0)
            return scores
        return generate_scores

    if mode == "seq_sum":
        def generate_scores(uid, sid, sess_items, item_pool):
            scores = Q[item_pool] @ Q[sess_items].sum(axis=0)
            return scores
        return generate_scores

    if mode == "no_seq":
        def generate_scores(uid, sid, sess_items, item_pool):
            scores = Q[item_pool] @ P[uid]
            return scores
        return generate_scores
## Main model ##


## Principal objective ##
def dense_Cui(Cui, cu):
    """ 
    To get dense representation of confidence matrix.
    
    Note: applicable only if your data is small!
    """
    return Cui.A + cu[:, np.newaxis]

def rmse_loss(local_factors, global_factors, Cui):
    """
    Simplified version of loss.
    """
    P, _ = local_factors
    Q, _ = global_factors
    n_users, n_items = Cui.shape
    overall_error = np.linalg.norm((P @ Q.T) - Cui._with_data(np.ones_like(Cui.data)), ord='fro')
    return overall_error / np.sqrt(n_users * n_items)

def main_objective(local_factors, global_factors, A, C, S_dict):
    """
    Objective that SeqMF model tries to optimize.

    Parameters
    ----------
    local_factors : tuple
        Tuple of local factors s.t. (P, W) where P is used for prediction

    global_factors : tuple
        Tuple of global factors s.t. (Q, Qb) where Q is used for prediction

    A : numpy.array
        Dense matrix of user preferences

    C : numpy.array
        Dense matrix of user confidence

    S_dict : dict of csr_matrices
        Dictionary of csr_matrices representing sequential dynamics of the data

    Returns
    -------
    err : float
        Overall loss
    """
    P, _ = local_factors
    Q, _ = global_factors

    _, n_items = C.shape
    e = np.ones(P.shape[1])
    err = 0.0
    for u in range(P.shape[0]):
        S = S_dict.get(u)
        if S is not None:
            seq = np.dot(np.multiply(S.dot(Q), Q), e)
        else:
            seq = np.zeros(n_items)
        res = A[u] - np.dot(Q, P[u]) + seq
        err += (C[u] * res) @ res
    return err
## Principal objective ##


## For new dynamic experiments ##
def update_p(
    P,
    Q,
    data,
    n_users,
    n_items,
    user_col,
    item_col,
    pow_bool,
    regularization,
    lap_smooth,
    gamma,
):
    """
    Update local factors(P) using only subset of users.
    """
    Cui, cu = get_conf_mtx_lap_smooth(
        data,
        n_users,
        n_items,
        user_col,
        item_col,
        lap_smooth,
        gamma,
    )
    assert Cui.shape == (n_users, n_items)

    S_dict = get_users_sui(
        data,
        n_items,
        user_col,
        item_col,
        pow_bool=pow_bool,
        level=user_col,
    )

    users = data[user_col].unique()
    _, n_factors = P.shape

    regI = np.diag(regularization * np.ones(n_factors))

    for u in users:
        P[u] = least_squares_P(Cui, cu, S_dict, Q, regI, u)

def get_grad_Q_partly(Q, C, cu, S_dict, P, reg, users):
    """
    Calculate gradient as a sum over some users represented in the data.
    """
    grad = reg * Q
    for user in users:
        grad += grad_update(Q, C, cu, S_dict, P, user)
    return grad

def update_Q_partly(
    Q,
    P,
    data,
    n_users,
    n_items,
    user_col,
    item_col,
    pow_bool,
    regularization,
    lap_smooth,
    gamma,
    lr,
    n_steps,
):
    """
    Update global factors(Q) using only subset of users.
    """
    Cui, cu = get_conf_mtx_lap_smooth(
        data,
        n_users,
        n_items,
        user_col,
        item_col,
        lap_smooth,
        gamma,
    )
    assert Cui.shape == (n_users, n_items)

    S_dict = get_users_sui(
        data,
        n_items,
        user_col,
        item_col,
        pow_bool=pow_bool,
        level=user_col,
    )

    users = data[user_col].unique()

    for _ in range(n_steps):
        logging.debug('l2-norm of global factors norm=%e', np.linalg.norm(Q))
        grad = get_grad_Q_partly(Q, Cui, cu, S_dict, P, regularization, users)
        Q -= lr * grad

    return Q
## For new dynamic experiments ##


## For dynamic experiments: dynamic_experiment.ipynb ##
def get_user_conf(data, item_col, n_items, alpha):
    """
    Confidence weights for a particular user.
    """
    items = data[item_col]
    vc = items.value_counts()
    conf = np.array([vc.get(x, 0) for x in range(n_items)]) + alpha
    return conf / conf.sum()

def updated_p(
    local_factors,
    global_factors,
    data,
    uidx,
    user_col,
    item_col,
    regularization,
    lap_smooth,
    pow_bool,
):
    """
    Inplace update of local factor for a particular user. 
    """
    Q = global_factors[0]
    n_items, n_factors = Q.shape
    au = get_user_pref(data, item_col)
    cu = get_user_conf(data, item_col, n_items, lap_smooth)
    Su = get_users_sui(data, n_items, user_col, item_col, pow_bool, user_col)[uidx]
    R = np.zeros((n_factors,)*2)
    R[np.diag_indices_from(R)] += regularization

    # calculate A
    A_sys = R + Q.T.dot(cu[:, np.newaxis] * Q)

    # seq
    if Su is None:
        seq = np.zeros(Q.shape[0])
    else:
        seq = np.diag(Su.dot(Q) @ Q.T)

    # calculate b
    b_sys = np.zeros(n_items)
    b_sys[au] = cu[au]
    b_sys -= cu * seq
    b_sys = Q.T @ b_sys
    local_factors[0][uidx] = np.linalg.solve(A_sys, b_sys)
## For dynamic experiments: dynamic_experiment.ipynb ##