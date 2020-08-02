import pandas as pd


class DataSource:

    def __init__(self,
                 name_id=None,
                 name_target=None,
                 name_csv_label='label',
                 name_csv_train='train',
                 name_csv_test='test',
                 name_csv_predict='predict'):
        '''
        Deal data from data sources
        :param id_name: String with unique id column name.
        :param target_name: String with target column name in train dataframe.
        :param name_csv_label: String with test target archive name without ".csv".
        :param name_csv_train: String with train archive name without ".csv".
        :param name_csv_test: String with test archive name without ".csv".
        :param name_csv_predict: String with predict archive name without ".csv".
        :return: DataSource object
        '''
        self.path_train = f'../data/{name_csv_train}.csv'
        self.path_test = f'../data/{name_csv_test}.csv'
        self.path_label = f'../data/{name_csv_label}.csv'
        self.path_predict = f'../data/{name_csv_predict}.csv'
        self.name_id = name_id
        self.name_target = name_target
        # TODO definir um seed padrão

    def read_data(self, is_train_stage=True):
        '''
            Read data from data sources
            :param etapa_treino: Boolean specifing if is train or test.
            :return: pd.DataFrame with values and pd.Series with labels
        '''
        return pd.read_csv(self.path_train if is_train_stage else self.path_test)

    def read_column(self, name_column=None, is_train_stage=True):
        if name_column is None : name_column = self.name_id
        return self.read_data(is_train_stage)[name_column]

    # TODO definir função para separar os dados de treino dos dados de teste Y
    #y = pd.read_csv(self.path_label)
        
    