# AutoGluon
import time

import numpy as np
import pandas as pd
from numba import njit, prange
#from sklearn.linear_model import RidgeCV
from autogluon.tabular import TabularPredictor

from models.time_series_models import TimeSeriesRegressor
from utils.tools import save_test_duration, save_train_duration

name = "AutoGluon"

def to_df(x: np.array):
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    return pd.DataFrame(x)

class AutoGluonRegressor(TimeSeriesRegressor):
    """
    This is a class implementing AutoGluon for time series regression.
    The code is adapted by the authors from the original Rocket implementation at https://github.com/awslabs/autogluon
    """

    def __init__(self,
                 output_directory: str):
        """
        Initialise the AutoGluon model

        Inputs:
            output_directory: path to store results/models
            n_kernels: number of random kernels
        """
        super().__init__(output_directory)
        print('[{}] Creating Regressor'.format(self.name))
        self.name = name
        self.regressor = TabularPredictor(label='class')

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            x_val: np.array = None,
            y_val: np.array = None):
        """
        Fit AutoGluon

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
        """
        start_time = time.perf_counter()
        print('[{}] Training'.format(self.name))
        #self.regressor.fit(x_train, y_train)
        # Reshape
        train = to_df(x_train)
        train['class'] = y_train # autogluon takes X and y in the same dataframe
        self.regressor.fit(train)
        self.train_duration = time.perf_counter() - start_time
        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)
        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

    def predict(self, x: np.array):
        """
        Do prediction with AutoGluon

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        # Reshape and cast
        x = to_df(x)
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()
        y_pred = self.regressor.predict(x)
        test_duration = time.perf_counter() - start_time
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
        print("[{}] Predicting completed, took {}s".format(self.name, test_duration))
        return y_pred
