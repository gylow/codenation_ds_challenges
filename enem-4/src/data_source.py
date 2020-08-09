import pandas as pd


class DataSource:

    def __init__(self,
                 name_id=None,
                 name_target=None,
                 rows_remove=None,
                 outliers_remove=None,
                 name_csv_label='label',
                 name_csv_train='train',
                 name_csv_test='test',
                 name_csv_predict='predict'):
        '''
        Deal data from data sources
        :param name_id: String with unique id column name.
        :param name_target: String with target column name in train dataframe.
        :param rows_remove: List of tuple(label, value_corresp) with the rows condictions to remove from Train data frame
        :param outliers_remove: List of tuple(label, value_corresp) with the rows condictions to remove from Train data frame
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
        self.rows_remove = rows_remove
        self.outliers_remove = outliers_remove
        self.name_id = name_id
        self.name_target = name_target
        # TODO definir um seed padrão

    def read_data(self, is_train_stage=True, original=False):
        '''
            Read data from data sources
            :param etapa_treino: Boolean specifing if is train or test.
            :param original: Boolean specifing if read original data frame or with removed rows 
            :return: pd.DataFrame with values and pd.Series with labels
        '''
        df = pd.read_csv(self.path_train if is_train_stage else self.path_test)

        if self.rows_remove and not original:
            for label, x in self.rows_remove :
                df = df[df[label] != x]

        if is_train_stage and self.outliers_remove:
            for label, x in self.outliers_remove :
                df = df[df[label] != x]

        return df

    def get_removed_rows(self, name_columns=None, is_train_stage=True):
        '''
            Read especifics columns from data sources
            :param name_columns: List with columns names
            :return: pd.DataFrame with especificated columns
        '''

        if self.rows_remove is None :
            return pd.DataFrame(columns=[name_columns])
            
        df = self.read_data(is_train_stage, original=True)
        df = pd.concat([df[df[label] == x] for label, x in self.rows_remove])

        return df if name_columns is None else df[name_columns]


    def get_columns(self, name_columns=None, is_train_stage=True):
        '''
            Read especifics columns from data sources
            :param name_columns: List with columns names
            :return: pd.DataFrame with especificated columns
        '''
        if name_columns is None : name_columns = self.name_id
        return self.read_data(is_train_stage)[name_columns]

    # TODO definir função para separar os dados de treino dos dados de teste Y
    #y = pd.read_csv(self.path_label)
        
    