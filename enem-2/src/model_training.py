import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load

from data_source import DataSource
from preprocessing import Preprocessing
from experiments import Experiments 


class ModelTraining:
    def __init__(self):
        self.data = DataSource()
        self.preprocessing = None
        # TODO separar dados de Teste dos dados de Treino
        
    def model_training(self):
        '''
        Train the model.
        :return: Dict with trained model, preprocessing used and columns used in training
        '''
        pre = Preprocessing()
        print('Loading data')
        df = self.data.read_data(etapa_treino = True)
        print('Training preprocessing')
        X_train, y_train = pre.process(df, etapa_treino = True)
        print('Training Model')
        model_obj = LinearRegression() # TODO linkar com a classe de experimentos para retornar o de melhor desempenho
        model_obj.fit(X_train, y_train)
        model = {'model_obj' : model_obj,
                 'preprocessing' : pre,
                 'colunas' : pre.feature_names }
        print(model)
        dump(model, '../output/modelo.pkl')
        return model
    
    