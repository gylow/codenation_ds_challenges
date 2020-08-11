import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
#from preprocessing import Preprocessing
#from data_source import DataSource
from metrics import Metrics


class Experiments:
    def __init__(self, regression=True):
        self.regression_algorithms = {'linear': LinearRegression(),
                                      'ridge': Ridge(),
                                      'lasso': Lasso(),
                                      'decision_tree': DecisionTreeRegressor(),
                                      'random_forest': RandomForestRegressor(),
                                      'svm': SVR(),
                                      'catboost': CatBoostRegressor()}
        self.classification_algorithms = {'decision_tree': DecisionTreeRegressor(),
                                          'random_forest': RandomForestRegressor(),
                                          'catboost': CatBoostRegressor()}
        self.dict_of_models = None
        self.regression = regression
        '''
        Choose the best algorithms to fit the problem
        :param regression: Boolean representing the training model: True for Regression or False for Classification
        '''
        # TODO implementar hiperparametros

    def get_model(self, alg):
        '''
        :param alg: String with the algorithm name to return
        :return: Algorithm class
        '''
        if self.regression:
            return self.regression_algorithms[alg]
        else:
            return self.classification_algorithms[alg]

    def train_model(self, x_train, y_train):
        '''
        Train the model with especified experiments
        :param x_train: pd.DataFrame with train data
        :param y_train: pd.Series with train labels
        :return: Dict with trained model
        '''

        algorithms = self.regression_algorithms if self.regression else self.classification_algorithms

        for alg in algorithms.keys():
            print('ALERT: Treinando o modelo ', alg)
            test = algorithms[alg]

            print(f"ALERT: {test}")
            test.fit(x_train, y_train)
            
            if self.dict_of_models is None:
                self.dict_of_models = {alg: test}
            else:
                self.dict_of_models.update({alg: test})
        return self.dict_of_models

    def run_experiment(self, df, y, test_size=0.15, seed=42):
        '''
        Run especified experiments
        :param df: Data Frame with features and target
        :param test_size: Float with percentage splited to test
        :param seed_random: Int with seed to random functions
        :return: Dataframe with all metrics
        '''

        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)
        
        models = self.train_model(x_train, y_train)
        
        df_metrics = pd.DataFrame()

        print('Running Metrics')
        for model in models.keys():
            print(f'ALERT: Predizendo os testes de {model}')
            y_pred = models[model].predict(x_test)
            print(f'ALERT: y : {y_pred}')

            metrics = Metrics().calculate_regression(y_test, pd.Series(y_pred))            
            df_metrics = df_metrics.append(pd.Series(metrics, name=model))

            pd.DataFrame.from_dict(metrics, orient='index').to_csv(
                '../output/'+model+'.csv')

        return df_metrics
