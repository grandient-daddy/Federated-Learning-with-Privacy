import logging

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from scipy.sparse import coo_matrix

from seqmf_pp import get_users_sui


class BaseGlobalModel:

    def __init__(self, fed: 'Federation'):
        self.fed = fed

    def fit(self, data: pd.DataFrame):
        raise NotImplementedError

    def predict(self, data: pd.DataFrame):
        """Method predict makes predictions for all users unlike local model
        which makes predictions only to specific user.
        """
        raise NotImplementedError


class BaseLocalModel:
    """Class BaseLocalModel represents a single model related to a single user
    in federation.

    :param fed: Federation state object.
    :param uid: Abstract user identifier in federation.
    """

    def __init__(self, fed: 'Federation', uid: Any):
        self.fed = fed
        self.uid = uid

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(uid={self.uid})'

    def fit(self, data: pd.DataFrame):
        raise NotImplementedError

    def predict(self, topk: int = 5):
        raise NotImplementedError


@dataclass
class Federation:

    # Low-frequency components of model (items x factors).
    global_factors: np.ndarray

    # High-frequency components of model (users x factors).
    local_factors: np.ndarray

    # (users).
    confidence: np.ndarray

    # (users x items).
    feedback: np.ndarray

    # Transition matrix between items (users x items x items).
    transitions: np.ndarray

    global_model_factory: Callable[..., BaseGlobalModel] = BaseGlobalModel

    local_model_factory: Callable[..., BaseLocalModel] = BaseLocalModel

    def __repr__(self) -> str:
        args = ', '.join([
            f'nofactors={self.nofactors}',
            f'noitems={self.noitems}',
            f'nousers={self.nousers}',
        ])
        return f'{self.__class__.__name__}({args})'

    @property
    def nofactors(self) -> int:
        """Property nofactors returns a number of factors used to represent
        users and items.
        """
        return self.global_factors.shape[1]

    @property
    def noitems(self) -> int:
        return self.global_factors.shape[0]

    @property
    def nousers(self) -> int:
        return self.local_factors.shape[0]

    @property
    def global_model(self) -> BaseGlobalModel:
        return self.global_model_factory(self)

    def local_model(self, uid: Any) -> BaseLocalModel:
        return self.local_model_factory(self, uid)

    @property
    def local_models(self) -> Iterable[BaseLocalModel]:
        for uid in range(self.nousers):
            yield self.local_model_factory(self, uid)


def random_federation(nofactors: int,
                      noitems: int,
                      nousers: int,
                      seed: Optional[int] = None) -> Federation:
    rng = np.random.RandomState(seed)
    fed = Federation(global_factors=rng.normal(size=(noitems, nofactors)),
                     local_factors=rng.normal(size=(nousers, nofactors)),
                     confidence=rng.uniform(size=(nousers, )),
                     feedback=rng.randint(0, 6, (nousers, noitems)),
                     transitions=rng.normal(size=(noitems, noitems)))
    return fed


def simulate(fed: Federation, dataset: pd.DataFrame, callbacks=None,
             full_history=False):
    """Function simulate simulates federated communications among local clients
    and global server.
    """

    callbacks = [*enumerate(callbacks)]

    def callback(*args, **kwargs):
        for i, (id, fn) in enumerate(callbacks):
            try:
                fn(*args, **kwargs)
            except Exception:
                logging.exception('callback #%d failed: skip it', id)
                del callbacks[i]

    # We assume that dataset has multi index. The first part of index
    # corresponds to global updates and the second one corresponds to local
    # updates. For example, the first key is a timestamp rounded up to 30 days
    # and the second on is timestamp rounded up to 1 day.
    for i, (supkey, group) in enumerate(dataset.groupby(level=[0])):
        for j, (subkey, frame) in enumerate(group.groupby(level=[1])):
            logging.info('suokey=%d subkey=%d', subkey, supkey)
            # Invoke all callbacks. Callback function can be used to evaluate
            # model as example.
            callback((supkey, subkey), frame, fed)

            if not full_history:
                # Update local models with only last batch of data.
                logs = frame \
                    .reset_index() \
                    .set_index(['uid', 'timestamp']) \
                    .groupby(level=[0])

                for uid, log in logs:
                    fed.local_model(uid).fit(log)
            else:
                # Update local models with full history.
                #
                # construction and accumulate statistics iteratively or
                # precomute all of them at the single point.
                last = frame.index[-1]
                logs = dataset \
                    .loc[:last] \
                    .reset_index() \
                    .set_index(['uid', 'timestamp']) \
                    .sort_index()

                for uid in fed.transitions:
                    fed.transitions[uid] = coo_matrix(
                        arg1=([], ([], [])),
                        shape=(fed.noitems, fed.noitems),
                    ).tocsr()

                fed.transitions.update(get_users_sui(
                    data=logs.reset_index(),
                    n_items=fed.noitems,
                    user_col='uid',
                    item_col='iid',
                    pow_bool=False,
                    level='uid',
                ))

                # parameters specific to each user in a single point. We should
                # estimate transition matrices inside fit method.
                #
                # >>> for uid, log in logs:
                # >>>     fed.local_model(uid).fit(log)

        # Update global model.
        fed.global_model.fit(dataset.loc[:supkey])

    return fed
