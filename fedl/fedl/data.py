import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet
import scipy as sp
import scipy.sparse

from dataclasses import dataclass
from json import loads
from pathlib import Path
from typing import Dict, Optional

from scipy.sparse import csr_matrix


__all__ = ('read_dataset', 'read_dataset_lsapp')


@dataclass
class Dataset:
    """Class Dataset is a container for session log and precalculated
    aggregates like feedback matrix or transition matrix.
    """

    sessions: pd.DataFrame

    transits: pd.DataFrame

    feedback: csr_matrix

    noitems: int

    nousers: int

    def __repr__(self):
        args = ', '.join([
            f'nousers={self.nousers}',
            f'noitems={self.noitems}',
            f'noevents={len(self.sessions)}',
        ])
        return f'Dataset({args})'


def make_feedback_matrix(data: pd.DataFrame, shape=None,
                         c0=1, gamma=1) -> csr_matrix:
    # Infer shape of feedback matrix if it is not given. In the code below, we
    # assume that users and items are zero-based indexed.
    if shape is None:
        nousers = data.user_id.max() + 1
        noitems = data.app_id.max() + 1
        shape = (nousers, noitems)

    # Aggregate coocurence statistics.
    counts = data \
        .set_index(['user_id', 'app_id']) \
        .groupby(level=[0, 1])[()] \
        .size() \
        .rename('value') \
        .to_frame() \
        .reset_index()

    # Prepare frequence exponents.
    freqs = sp.sparse \
        .coo_matrix((counts.value, (counts.user_id, counts.app_id)), shape) \
        .tocsr() \
        .power(gamma)

    # Uniform scaling factor across users.
    scales = np.ones(shape[0]) / c0

    # Prepare scaling and normalization matrix.
    norm = freqs.sum(axis=1).A.T[0].astype(float)
    norm_mask = np.nonzero(norm != 0)
    norm[norm_mask] = scales[norm_mask] / norm[norm_mask]  # c^0_u / Z_u
    norm = sp.sparse.diags(norm)

    feedback = norm @ freqs
    return feedback


def make_transition_matrix(ts: pd.Series, shape=None,
                           drop_last=False) -> csr_matrix:
    """Function make_transition_matrix makes transition matrix from timeseries
    data.
    """
    if shape is None:
        size = ts.values.max() + 1
        shape = (size, size)

    prevs = ts.values[:-1]
    nexts = ts.values[1:]
    if drop_last:
        prevs = prevs[:-1]
        nexts = nexts[:-1]

    mat = sp.sparse \
        .coo_matrix((np.ones_like(prevs), (prevs, nexts)), shape) \
        .tocsr()

    return mat


def make_transition_matrices(data: pd.DataFrame,
                             size: Optional[int] = None) -> csr_matrix:
    """Function make_transition_matrices prepares transition matrix for each
    unique index entry. By default, it assume that input data are indexed with
    `user_id` and `session_id` fields.
    """
    if size is None:
        size = data.app_id.max() + 1

    mats = data \
        .set_index(['user_id', 'session_id', 'timestamp'])['app_id'] \
        .sort_index() \
        .groupby(level=[0, 1]) \
        .apply(lambda x: make_transition_matrix(x, shape=(size, size))) \
        .rename('transition_matrix') \
        .reset_index()

    return mats


def read_dataset_lsapp(root_dir: Path,
                       parts=('train', 'test')) -> Dict[str, Dataset]:
    logging.info('load datasets for label(s): %s', ', '.join(parts))
    datasets = {}
    for part in parts:
        logging.info('[%5s] read session log from filesystem', part)
        path = root_dir / f'lsapp.{part}.parquet'
        tbl = pa.parquet.read_table(path)
        sessions = tbl.to_pandas()
        noitems = loads(tbl.schema.metadata[b'noitems'])
        nousers = loads(tbl.schema.metadata[b'nousers'])

        logging.info('[%5s] estimate transition matrices', part)
        transits = make_transition_matrices(sessions, noitems)

        logging.info('[%5s] estimate feedback matrix', part)
        shape = None
        if nousers is not None and noitems is not None:
            shape = (nousers, noitems)
        feedback = make_feedback_matrix(sessions, shape)

        datasets[part] = Dataset(sessions, transits, feedback, noitems,
                                 nousers)

    return datasets


def read_dataset(name, root_dir) -> Dict[str, Dataset]:
    if name == 'lsapp':
        return read_dataset_lsapp(Path(root_dir), ['train', 'val', 'test'])
    else:
        raise ValueError(f'Unexpected dataset name: {name}.')
