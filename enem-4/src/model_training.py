import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from experiments import Experiments
from joblib import dump, load

#from data_source import DataSource
#from preprocessing import Preprocessing


class ModelTraining:
    def __init__(self, preprocessing, regression=True, seed=42):
        self.pre = preprocessing
        self.regression = regression
        self.seed = seed
        '''
        :param preprocessing: Preprocessing object
        :param regression: Boolean representing the training model: True for Regression or False for Classification
        :param seed: Int with seed to random functions
        '''

    def training(self):
        '''
        Train the model.
        :return: Dict with trained model, preprocessing used and columns used in training
        '''

        print('Training preprocessing')
        df_train, y = self.pre.process()

        print('Training Model')
        exp = Experiments(regression=self.regression)
        df_metrics = exp.run_experiment(df_train, y, seed=self.seed)
        print('Metrics:', df_metrics)

        alg_better = df_metrics[df_metrics.r_2_score ==
                                df_metrics.r_2_score.max()].index[0]
        print('ALERT: chosen algorithm: ',alg_better)

        model_obj = exp.get_model(alg_better)
        model_obj.fit(df_train, y)
        model = {'model_obj': model_obj,
                 'preprocessing': self.pre,
                 'colunas': self.pre.get_name_features()}

        # print(model)
        dump(model, '../output/model.pkl')

        return model
