import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

#from data_source import DataSource
#from preprocessing import Preprocessing
#from experiments import Experiments


class ModelTraining:
    def __init__(self, preprocessing, model_training=None):
        self.pre = preprocessing
        self.model_training = model_training
        # TODO separar dados de Teste dos dados de Treino
        '''
        :param preprocessing: Preprocessing object
        :param model_training: String with model training type: 'LR' or 'RF' for LinearRegression or RandomForestRegressor
        '''

    def training(self):
        '''
        Train the model.
        :return: Dict with trained model, preprocessing used and columns used in training
        '''

        print('Training preprocessing')
        X_train, y_train = self.pre.process()

        #print(f'Y_train : \n {y_train}\n')
        #print(f'X_train : \n {X_train}\n')

        print('Training Model')
        # TODO linkar com a classe de experimentos para retornar o algoritmo de melhor desempenho
        model_obj = RandomForestRegressor() if self.model_training == 'RF' else LinearRegression()
        model_obj.fit(X_train, y_train)
        model = {'model_obj': model_obj,
                 'preprocessing': self.pre,
                 'colunas': self.pre.get_feature_names()}

        # print(model)
        dump(model, '../output/modelo.pkl')

        return model
