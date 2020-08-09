import pandas as pd
import numpy as np
from joblib import dump, load

#from model_training import ModelTraining
#from metrics import Metrics
# rom preprocessing import Preprocessing
#from data_source import DataSource


class ModelInference:
    def __init__(self, model=None):
        self.modelo = load('../output/modelo.pkl') if model is None else model
        # TODO implementar classe de testes unitários
        '''
        :param model: Model_training object training return
        '''

    def predict(self):
        '''
        Predict values using model trained.
        :return: pd.Series with predicted values.
        '''

        print('Preprocessing Data')
        X_test = self.modelo['preprocessing'].process(is_train_stage=False)

        # Redundante, modelo não funciona com NA
        # print(f'Quantidade de NA: {X_test.isna().sum()}')

        print('Predicting')
        # TODO verificar mudanças no contexto dos dados de produção
        y_pred = self.modelo['model_obj'].predict(X_test)

        print('Saving Files')
        pd.DataFrame(y_pred).to_csv(self.modelo['preprocessing'].data.path_predict)

        return y_pred
