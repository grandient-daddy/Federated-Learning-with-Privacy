import numpy as np

from seqmf_pp import (
    MethodError,
    adam,
    get_conf_mtx_lap_smooth,
)

## Local factors update ##
def build_linear_equation_P(C, cu, Q, R, u):
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

    # calculate b
    QtC = Qnnz.T * coef
    b_sys = QtC.sum(axis=1) + Qcu[inds].T.sum(axis=1)
    return A_sys, b_sys

def least_squares_P(C, cu, Q, R, u):
    """
    The function produces local factor for user 'u' using ALS.
    """
    A_sys, b_sys = build_linear_equation_P(C, cu, Q, R, u)
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

def update_P(P, C, cu, Q, regI):
    """
    Inplace ALS update of local factors(P)
    """
    n_users, _ = C.shape
    for u in range(n_users):
        P[u] = least_squares_P(C, cu, Q, regI, u)
## Local factors update ##


## Global factors update ##
def grad_update(Q, C, cu, P, u):
    """
    The function produces gradient related to user 'u'.
    """
    indptr = C.indptr
    inds = C.indices[indptr[u]:indptr[u + 1]]
    coef = C.data[indptr[u]:indptr[u + 1]]

    p_u = P[u]
    r_u = Q.dot(p_u)

    r_u[inds] -= 1
    r_u_const = r_u * cu[u]
    r_u[inds] *= coef 
    Du = r_u + r_u_const
    grad = np.outer(Du, p_u)
    return grad

def get_grad_Q(Q, C, cu, P, reg):
    """
    The function produces gradient related to the sum of all the users in a system
    """
    grad = reg * Q
    n_users, _ = C.shape
    for user in range(n_users):
        grad += grad_update(Q, C, cu, P, user)
    return grad

def update_Q(Q, C, cu, P, lr, reg, n_steps, adam_config):
    """
    Inplace Gradient Descent(ADAM) update of global factors(Q)
    """
    for _ in range(n_steps):
        grad = get_grad_Q(Q, C, cu, P, reg)
        if adam_config:
            grad, adam_config['cum_grad'], adam_config['cum_sq_grad'] = adam(grad, **adam_config)
        Q -= lr * grad
## Global factors update ##
    

## Main model ##
def mf(
    data,
    n_users,
    n_items,
    user_col,
    item_col,
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
    pow_bool=None,
):

    """
    Matrix Factorization.

    An implicit Matrix Factorization Recommender System using hybrid training strategy:
    ALS for local factors P, GD(Adam) for global factors Q.
    
    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe that has at least 2 columns related to object and subject, e.g. item and user

    n_users : int
        The number of unique subjects(users) in the system

    n_items : int
        The number of unique objects(items) in the system

    user_col : str
        Name of subject column in the 'data' Dataframe

    item_col : str
        Name of object column in the 'data' Dataframe

    n_factors : int
        The dimensionality of a latent space

    n_epochs : int
        The number of full updating cycles

    regularization : float
        The Tikhonov regularization constant

    lap_smooth : float
        alpha parameter of laplacian smoothing

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

    pow_bool: None
        For compatability.

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
    assert Cui.shape == (n_users, n_items)

    #Initialize factors:
    random_state = np.random.RandomState(seed)
    P = random_state.normal(0, 0.01, size=(n_users, n_factors))
    Q = random_state.normal(0, 0.01, size=(n_items, n_factors))

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
            update_P(P, Cui, cu, Q, regI)
        except MethodError as err:
            print("Error:", err.params)
            return (np.zeros_like(P), None), (np.zeros_like(Q), None)

        if evaluation_callback:
            evaluation_callback((P, None), (Q, None))

        # update Q:
        update_Q(Q, Cui, cu, P, lr, regularization, n_steps, adam_config)
        
        if evaluation_callback:
            evaluation_callback((P, None), (Q, None))
        
    return (P, None), (Q, None)

def get_scores_generator(local_factors, global_factors, mode="mf", pow_bool=None):
    """
    Creates scores generator for the provided local and global factors.

    Parameters
    ----------
    local_factors : tuple
        Tuple of local factors s.t. (P, W) where P is used for prediction
    
    global_factors : tuple
        Tuple of global factors s.t. (Q, Qb) where Q is used for prediction
    
    mode : str, optional
        The recommendation mode has the following options:
        - 'mf': Does not take into account sequential information, only long term preferences
        Default is 'mf'.

    pow_bool: None
        For compatability.

    Returns
    -------
    generate_scores : function
        Function to generate scores for a particular user, session, items.
    """
    P, _ = local_factors
    Q, _ = global_factors

    if mode == "mf":
        def generate_scores(uid, sid, sess_items, item_pool):
            scores = Q[item_pool] @ P[uid]
            return scores
        return generate_scores
## Main model ##

## Principal objective ##
def dense_Cui(Cui, cu):
    return Cui.A + cu[:, np.newaxis]

def main_objective(local_factors, global_factors, A, C):
    """
    Objective SeqMF++ model tries to optimize.

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

    Returns
    -------
    err : float
        Overall loss
    """
    P, _ = local_factors
    Q, _ = global_factors
    err = 0.0
    for u in range(P.shape[0]):
        res = A[u] - np.dot(Q, P[u])
        err += (C[u] * res) @ res
    return err
## Principal objective ##