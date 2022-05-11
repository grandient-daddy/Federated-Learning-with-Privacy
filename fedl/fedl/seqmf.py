import numpy as np
import pandas as pd

from seqmf_pp import seq_mf_pp


class SeqMFPlusPlus:
    """Class SeqMFPlusPlus implements SeqMF++ algorithm for sequence-aware
    matrix factorization based algorihm for recommender systems.

    :param alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no
                  smoothing).
    """

    def __init__(
        self,
        n_factors: int,
        n_items: int,
        n_users: int,
        alpha=5e-1,
        gamma=5e-1,
        regularization=1e-1,
        learning_rate=5e-5,
        n_epochs=6,
        n_steps=4,
        pow_bool=False,
        optimizer_mode='sgd',
        random_state=None,
    ):
        self.n_factors = n_factors
        self.n_items = n_items
        self.n_users = n_users
        self.alpha = alpha
        self.gamma = gamma
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.pow_bool = pow_bool
        self.random_state = random_state or np.random.RandomState()

        # Forward declaration for model state.
        self.global_factors: np.ndarray
        self.local_factors: np.ndarray

    def __repr__(self) -> str:
        args = ', '.join([
            f'n_factors={self.n_factors}',
            f'n_items={self.n_items}',
            f'n_users={self.n_users}',
        ])
        return f'{self.__class__.__name__}({args})'

    def fit(self, X: pd.DataFrame):
        """Method fit fits model in offline regine when all data is availiable.
        It assumed that item column and user column are named 'iid' and 'uid'
        respectively.
        """
        local_factors, global_factors = seq_mf_pp(
            data=X,
            n_users=self.n_users,
            n_items=self.n_items,
            user_col='uid',
            item_col='iid',
            pow_bool=self.pow_bool,
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            regularization=self.regularization,
            lap_smooth=self.alpha,
            gamma=self.gamma,
            lr=self.learning_rate,
            n_steps=self.n_steps,
            seed=self.random_state,  # TODO Use either seed or random state.
            evaluation_callback=None,
            iterator=range,
            optimizer_mode='sgd',
        )

        self.global_factors = global_factors[0]
        self.local_factors = local_factors[0]
