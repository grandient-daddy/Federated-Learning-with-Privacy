import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

from dataprep import drop_consequtive_repeats, assign_session_id, list_session_lists

class DataProcessor:
    """
    Data preprocessing class for the next session-based evaluation.

    Parameters
    ----------
    file_name : str
        Name of the data file.

    column_names : str
        Comma-separated column names, e.g. 'userid,appid,timestamp' 
    
    userid : str, optional
        Name of the column representing users.
        Default is 'userid'.
    
    itemid : str, optional
        Name of the column representing items.
        Default is 'appid'.
    
    timeid : str, optional
        Name of the column representing time.
        Default is 'timestamp'.
    
    session_break_delta: str, optional
        Set session break e.g. '15min'. 
        It represents how the data is divided into sessions. The next session starts if 
        during session_break_delta time there was no interactions.
        Default is '30min'

    Attributes
    ---------- 
    userid : str
        Name of the column representing users.

    itemid : str
        Name of the column representing items.

    timeid : str
        Name of the column representing time.

    indexers : 
        Mapping into internal indices.

    n_users : int
        The number of unique users.

    n_items : int
        The number of unique items.

    seen_interactions : scipy.sparse.csr_matrix
        User/Item binary preference matrix.
    """

    def __init__(self, file_name, column_names, userid='userid', itemid='appid', timeid='timestamp', session_break_delta='30min'):
        self.file = file_name
        self.names = column_names
        self.session_break_delta = session_break_delta
        self.userid = userid
        self.itemid = itemid
        self.timeid = timeid
        self.indexers = None
        self.n_users = None
        self.n_items = None
        self.data = None
        self.train, self.valid, self.test = None, None, None
        self.seen_interactions = None


    def read_data(self, usecols=None, to_datatime_bool=True):
        """
        Read the data from self.file into dataframe.
        
        Parameters
        ----------
        usecols : list
            List of column names to leave in a dataframe.
        
        to_datatime_bool : bool, optional
            To transform or not(already transformed) self.timeid column into
            datatime format.
            Default is True.

        Returns
        -------
        data : pamdas.Dataframe
            Dataframe sorted by time and userid.
        """
        names = self.names.lower().split(',')
        data = (
            pd.read_csv(
                self.file, header=None, sep=' ',
                names=names, usecols=usecols,
                dtype={self.timeid: str}
            )
        )

        if to_datatime_bool:
            data = data.assign(timestamp=lambda x: pd.to_datetime(x[self.timeid], format="%Y%m%d%H%M%S"))

        data = data.sort_values([self.userid, self.timeid])
        return data


    def drop_conseq_repeats(self, data, window="1s"):
        """
        Drops all consecutive repeated events except the first one.
        If window is defined, repeated events outside the time window
        will be preserved, not dropped.
        """
        return data.pipe(drop_consequtive_repeats, window=window)


    def create_sessid_column(self, data):
        """
        Assigns global and local session ids based on self.session_break_delta
        """
        return (
            data
            .pipe(assign_session_id, session_break_delta=self.session_break_delta)
            .assign(
                sessid_global = lambda x: (x['timestamp'].diff() > pd.Timedelta(self.session_break_delta)).cumsum()
            )
        )


    def _time_split_mask(self, cond):
        """
        Splits by time.
        Condition is checked on entire sessions, i.e.,
        it's met only if all session elements satisfy condition.
        """
        def splitter(df):
            return cond(df['timestamp']).groupby([df['userid'], df['sessid']]).transform('all')
        return splitter


    def _idx2ind(self, data):
        """
        Transform data indices into internal ones based on self.indexers
        """
        indices = uind, iind = [idx.get_indexer_for(data[idx.name]) for idx in self.indexers]
        dropped_inds = (uind == -1) | (iind == -1) # useful to keep for e.g. cold-start experiments
        checked_inds = ~dropped_inds
        reindexed_data = (
            data
            .loc[checked_inds]
            .assign(**{
                idx.name: ind[checked_inds] for ind, idx in zip(indices, self.indexers)
            })
        )
        return reindexed_data, dropped_inds


    def train_valid_test(self, data, reindex=True, test_interval='1d', valid_interval='1d'):
        """
        Splits data into train/validation/test based on time intervals.
        """
        test_start_time = data[self.timeid].max() - pd.Timedelta(test_interval)
        valid_start_time = test_start_time - pd.Timedelta(valid_interval)

        train_data = data.loc[self._time_split_mask(lambda x: x < valid_start_time)]
        test_data = data.loc[self._time_split_mask(lambda x: x >= test_start_time)]
        valid_data = data.loc[self._time_split_mask(lambda x: (x >= valid_start_time) & (x < test_start_time))]

        uidx_cat = train_data[self.userid].astype('category').cat
        iidx_cat = train_data[self.itemid].astype('category').cat
        self.indexers = [
            uidx_cat.categories.rename(self.userid),
            iidx_cat.categories.rename(self.itemid)
        ]
        self.n_users, self.n_items = map(len, self.indexers)

        if reindex:
            new_indices = {
                self.userid: uidx_cat.codes,
                self.itemid: iidx_cat.codes
            }
            train_data = train_data.assign(**new_indices)
            valid_data, valid_dropped = self._idx2ind(valid_data)
            test_data, test_dropped = self._idx2ind(test_data)
        return train_data, valid_data, test_data


    def split_by_column(self, data, column_name, reindex=True):
        """
        Splits data into train/validation/test based on column with "column_name".
        Corresponding column should have values: "train", "valid", "test" to be processed.
        """
        train_data = data[data[column_name] == "train"]
        valid_data = data[data[column_name] == "valid"]
        test_data = data[data[column_name] == "test"]

        uidx_cat = train_data[self.userid].astype('category').cat
        iidx_cat = train_data[self.itemid].astype('category').cat
        self.indexers = [
            uidx_cat.categories.rename(self.userid),
            iidx_cat.categories.rename(self.itemid)
        ]
        self.n_users, self.n_items = map(len, self.indexers)

        if reindex:
            new_indices = {
                self.userid: uidx_cat.codes,
                self.itemid: iidx_cat.codes
            }
            train_data = train_data.assign(**new_indices)
            valid_data, valid_dropped = self._idx2ind(valid_data)
            test_data, test_dropped = self._idx2ind(test_data)
        return train_data, valid_data, test_data


    def get_seen_interactions(self, data):
        """
        Get user/item preference matrix in scipy.sparse.csr_matrix format.
        Note: data is assumed to be already reindexed!
        """
        shape = (self.n_users, self.n_items)
        rows, cols = data[self.userid], data[self.itemid]
        seen = (coo_matrix((np.ones_like(rows), (rows, cols)), shape=shape, dtype=np.int32) > 0).astype(np.int32)
        return seen


    def prepare_data(self, usecols, test_interval, valid_interval, window=0, min_sess_length=2):
        """
        Prepare the data for the experiment:
        - Split into train/validation/test
        - Create preference matrix
        - Create session lists for train/validation/test
        
        Parameters
        ----------
        usecols : list
            List of column names to leave in a dataframe.
        
        test_interval : str
            The number of days to put into test e.g. '14d'
        
        valid_interval : str
            The number of days to put into validation e.g. '7d'

        window : object, optional
            Window allows to transform the data s.t. there are less repeated events.
            If window is defined e.g. '3s', repeated events outside the time window
            will be preserved, not dropped.
            Default is 0 and it means not to drop repeats in the session.

        min_sess_length : int, optional
            Minimal session length.
            Default is 2.

        Note: the function does not return the data but creates new objects as attributes of an object of 
        DataProcessor class:
            - self.data -> dataframe with raw data
            - self.train, self.valid, self.test -> splits of self.data
            - self.seen_interactions -> user\item preference matrix
            - self.train_sessions -> Pandas.Series of users' sessions
            - self.valid_sessions -> Pandas.Series of users' sessions
            - self.test_sessions -> Pandas.Series of users' sessions
        """
        # Read the data and sort by "userid" and "timestamp"
        self.data = self.read_data(usecols=usecols)
        if window != 0:
            self.data = self.drop_conseq_repeats(self.data, window=window)
        # Create "sessid" column
        self.data = self.create_sessid_column(self.data)

        # Divide data into train/validation/test and transform user/item id to internal indexes
        self.train, self.valid, self.test = self.train_valid_test(
            self.data,
            reindex=True,
            test_interval=test_interval,
            valid_interval=valid_interval,
        )

        self.seen_interactions = self.get_seen_interactions(self.train)

        self.train_sessions = list_session_lists(self.train)
        self.valid_sessions =  list_session_lists(
            self.valid, min_length=min_sess_length
        )
        self.test_sessions =  list_session_lists(
            self.test, min_length=min_sess_length
        )