# Team C

import time
import numpy as np
from numba import njit, prange
from models.time_series_models import TimeSeriesRegressor
from utils.tools import save_test_duration, save_train_duration
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
import os
from sys import path

name = "TeamC"

def to_df(x: np.array):
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    return pd.DataFrame(x)

class TeamCRegressor(TimeSeriesRegressor):
    """
    This is a class implementing a model for time series regression.
    """

    def __init__(self,
                 output_directory: str):
        """
        Initialise the model

        Inputs:
            output_directory: path to store results/models

        """
        super().__init__(output_directory)
        print('[{}] Creating Regressor'.format(self.name))
        self.name = name
        #self.regressor = RidgeCV(alphas=np.logspace(-3, 3, 10),
        #                         normalize=True)
        self.missing_value_holder = -1.111
        #self.models = {x: LGBMRegressor(learning_rate=0.25, random_state=2020) for x in self.output_label}
        self.model = LGBMRegressor(learning_rate=0.25, random_state=2020)
        # stacking model
        #self.model_stacking = MultiOutputRegressor(Ridge(alpha=1, random_state=2020))
        #clustering model
        self.n_clusters = 16
        self.clustering_model = KMeans(n_clusters=self.n_clusters)
        self.clustering_models = dict()
        #flights information
        self.vol_missing = []
        self.vol_complete = []
        # Features
        self.feat_selected = None
        self.feature_target = None
        self.is_trained = False

    def generate_cluster_features(self, df, cluster_name, cluster_features, training):
        """Generate cluster based on a feature and create features based on them

        Parameters
        ----------
        df : a pandas Dataframe
            A pandas DataFrame where each columns are sensors or gauges data and each row correspond to a time point.
        cluster_name : a string
            The column name of the resulting feature
        cluster_features : a string
            The column name of the data to uses to compute the clusters
        training : bool
            A flag indicating if the model should be trained first

        Returns
        -------
        A pandas DataFrame
            The DataFrame df with the additional columns "cluster_name"
        """
        # clustering model
        clustering_model = KMeans(n_clusters=self.n_clusters)
        if training:
            clustering_model.fit(df[cluster_features])
            self.clustering_models.update({cluster_name: clustering_model})
            self.feat_selected += [cluster_name]
        df.loc[:, cluster_name] = self.clustering_models[cluster_name].predict(
            df[cluster_features]
        )
        ## Show number of sample in every cluster
        if training:
            print(
                df[["vol_num", cluster_name]]
                .groupby(cluster_name)
                .count()
                .rename(columns={"vol_num": "Number"})
            )
        return df

    def feature_engineering(self, df, training=True):
        """Compute the needed feature, reduce the redundancy and format the data

        Parameters
        ----------
        df : a pandas DataFrame
            A DataFrame containing the sensors and the gauges of the flights
        training : bool, optional
            A flag indicating if the clustering model should be trained, by default True

        Returns
        -------
        A pandas DataFrame
            A DataFrame containing the formated and prepared data
        """
        ################### Add cluster number as features ###################
        df = self.generate_cluster_features(df, cluster_name, cluster_features, training)
        df = self.generate_cluster_features(df, cluster_name, cluster_features, training)
        return df

    def post_treatment(self, df_pred, flight):
        """The function performing the post processing of the data. It is used corrected the data on the flights with anomaly.

        Parameters
        ----------
        df_pred : a pandas DataFrame
            A dataframe containing the predicted gauges
        flight : A dbflight object
            An object containing the raw flight data

        Returns
        -------
        A pandas DataFrame
            A DataFrame containing the post processed gauges
        """
        flight_name = flight.name
        flight_num = int(flight_name.split(".")[0])

        ## Stacking prediction
        #df_pred.loc[:, self.output_label_2] = self.model_stacking.predict(
        #        df_pred.loc[:, self.output_label_2])
        #return df_pred
        return None

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            x_val: np.array = None,
            y_val: np.array = None):
        """
        Fit model

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
        """
        start_time = time.perf_counter()
        print('[{}] Training'.format(self.name))
        df_all = to_df(x_train)
        #df_y_pred = to_df(y_train)
        #df_all = self.feature_engineering(df_all) # ad-hoc clustering
        self.feat_selected = df_all.columns

        print(f"------------------- Fit! -------------------")
        print(f"Num feat: {len(self.feat_selected)}")
        self.model.fit(df_all[self.feat_selected], y_train)

        #df_y_pred = self.model.predict(df_all[self.feat_selected])
        #self.global_feature_importance()
        ################### Fit stacking models ###################
        #self.model_stacking.fit(df_y_pred[self.output_label_2], df_all[self.output_label_2])

        # tser
        #self.regressor.fit(x_train, y_train)
        self.train_duration = time.perf_counter() - start_time
        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)
        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))
        self.is_trained = True

    def predict(self, x: np.array):
        """
        Do predictions

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()
        df = to_df(x)
        #df = self.feature_engineering(df, training=False)

        ## Prediction
        y_pred = self.model.predict(df[self.feat_selected])

        # tser
        #y_pred = self.regressor.predict(x)
        test_duration = time.perf_counter() - start_time
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
        print("[{}] Predicting completed, took {}s".format(self.name, test_duration))
        return y_pred
