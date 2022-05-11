import matplotlib.pyplot as plt
from evaluation import evaluate

def find_optimal_params(
    model,
    train,
    valid_sessions,
    seen_interactions,
    grid,
    param_names,
    fixed_params,
    get_scorer,
    model_name='',
    target_metric='hr',
    metric_k=5,
    show_result=False,
    main_iterator=range
):
    """
    Find optimal parameters of a model and save them to dictionary.
    Note that the function saves only the results(score:params) that are better 
    then the previous best result according to score for the target metric.
    
    Parameters
    ----------
    model : function
        Model from seqmf_pp.py -> seq_mf_pp of mf.py -> mf
        
    train : pandas.Dataframe
        Interactions Dataframe that has at least 2 columns related to object and subject,
        e.g. item and user.
    
    valid_sessions : pandas.Series
        Data structure representing users' sessions s.t.
        valid_session[userid] = [[item1, item3, ...], [item1, ..., item6], [item3, ...], ...]

    seen_interactions : scipy.sparse.csr_matrix
        User/Item binary preference matrix.

    grid : list
        List of configurations to check.

    param_names : tuple
        Tuple of all the parameters names.

    fixed_params : dict
        Dictionary of fixed parameters, e.g. 
        {"n_users": n_users, "n_items": n_items, "user_col": user_col, "item_col": item_col,
         "optimizer_mode": optimizer_mode, "n_epochs": 4, 'n_steps': 4, "seed": 13,
         "evaluation_callback": None, "iterator": range}
    
    get_scorer : function
        get_scores_generator function from seqmf_pp.py or mf.py

    model_name : str, optional
        Name of a model, e.g. "SeqMF", "MF"
        Default is ''

    target_metric : str, optional
        Target quality metric: "hr", "mrr", "ndcg"
        Default is "hr"

    metric_k : int, optional
        'k' for metric@k.
        Default is 5

    show_result : bool, optional
        Show the results to standard output or not.
        Default is False

    main_iterator : object, optional
        range or ipypb.irange
        Default is range

    Returns
    -------
    results_dict : dict
        Dictionary with score and optimal parameters for the particular score,
        e.g. results_dict[0.865] = {'pow_bool': None, 'lap_smooth': 0.001, ...}
    """
    best_score = -1
    results_dict = {}
    for i in main_iterator(len(grid)):    
        model_conf = dict(zip(param_names, grid[i]))
        gen_score_mode = model_conf.pop("mode")

        local_factors, global_factors = model(
            train,
            **fixed_params,
            **model_conf,
        )

        up_generate_scores = get_scorer(
            local_factors,
            global_factors,
            mode=gen_score_mode,
            pow_bool=model_conf["pow_bool"]
        )

        metrics_df, _ = evaluate(
            up_generate_scores,
            valid_sessions,
            seen_interactions,
        )

        valid_results = (
            metrics_df
            .reset_index()
            .groupby(["topk"])
            .mean()[["hr", "mrr", "ndcg"]]
        )

        score = valid_results[target_metric][metric_k]

        if score > best_score:
            best_score = score
            results_dict[score] = dict(zip(param_names, grid[i]))
            if show_result:
                print(
                    f"{model_name}:"
                    + f"\nBest {target_metric}@{metric_k}: {score}"
                    + f"\nThe best performance parameters:"
                    + f'\n{results_dict[score]}'
                ) 
    return results_dict

def print_metrics(x, stat, model_name, n_factors, reg, figsize=(18, 10), title_fontsize=14):
    """
    Function to plot metric scores.
    
    Parameters
    ----------
    x : numpy.ndarray
        Range of x values.

    stat : dict
        Dictionary of the following structure:
        stat["metric"][top_k] = list()
    
    model_name : str
        Name of the model to put into title.

    n_factors : int
        The number of factors you want to put on suptitle.

    reg : float
        Regularization parameter you want to put on suptitle.

    figsize : tuple, optional
        Size of a figure.
        Default is (18, 10)

    title_fontsize : int, optional
        Size of a title text.
        Default is 14
    """
    plt.rcParams["figure.figsize"] = figsize
    stat_metrics = list(stat.keys())[:-1]
    topk = stat[stat_metrics[0]].keys()

    fig, axs = plt.subplots(3, 3)
    fig.suptitle(f"{model_name}. Metrics: rank_{n_factors}_reg_{reg}", fontsize=title_fontsize)
    for i, metric in enumerate(stat_metrics):
        for j, k in enumerate(topk):
            axs[i, j].plot(x, stat[metric][k], "-r*")
            axs[i, j].set_title(f"{metric}@{k}")
            axs[i, j].grid(True)
    fig.set_facecolor('white')

def print_norms(x, stat, model_name, figsize=(18, 4), title_fontsize=14):
    """
    Function to plot model norms.
    
    Parameters
    ----------
    x : numpy.ndarray
        Range of x values.

    stat : dict
        Dictionary of the following structure:
        stat["norms"]["factor_name"] = list()
    
    model_name : str
        Name of the model to put into title.

    figsize : tuple, optional
        Size of a figure.
        Default is (18, 4)

    title_fontsize : int, optional
        Size of a title text.
        Default is 14
    """
    plt.rcParams["figure.figsize"] = figsize
    model_metrics = stat["norms"].keys()

    fig, axs = plt.subplots(1, 4)
    fig.suptitle(f"{model_name}. Factor norms", fontsize=title_fontsize)
    for j, name in enumerate(model_metrics):
        axs[j].plot(x, stat["norms"][name])
        axs[j].set_title(f"Norm {name}")
        axs[j].grid(True)
    fig.set_facecolor('white')

def print_objective_scores(x, y, model_name, color="r-*", figsize=(18, 4), title_fontsize=14):
    """
    Function to plot objective scores.
    
    Parameters
    ----------
    x : numpy.ndarray
        Range of x values.

    y : numpy.ndarray
        Range of y values.
    
    model_name : str
        Name of the model to put into title.

    color : str, optional
        String with color specifics.
        Default is "r-*"

    figsize : tuple, optional
        Size of a figure.
        Default is (18, 4)

    title_fontsize : int, optional
        Size of a title text.
        Default is 14
    """
    plt.rcParams["figure.figsize"] = figsize
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y, color)
    axs.set_title(f"{model_name}. Loss.", fontsize=title_fontsize)
    axs.grid(True)
    fig.set_facecolor('white')