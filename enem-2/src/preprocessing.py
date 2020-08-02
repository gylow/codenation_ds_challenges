import category_encoders as ce
import pandas as pd
#from data_source import DataSource
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, data):
        self.feature_names = None
        self.categoric_features = None
        self.numeric_features = None
        self.scaler = None
        self.catb= None
        self.data = data
        '''
        Class for preprocessing data model.
        :param data: DataSource object
        :return: Preprocessed object
        '''

    def process(self, is_train_stage=True,
                missing_values_acceptable=0,
                unique_values_acceptable=100):
        '''
        Process data for training the model.
        :param etapa_treino: Boolean
        :param missing_values_acceptable: Int with minimum missing values acceptable percentage
        :param unique_values_acceptable: Int with maximum unique values acceptable percentage
        :return: processed Pandas Data Frame
        '''
        df = self.data.read_data(is_train_stage)
        percentage = 100/df.shape[0]

        print('Creating DataFrame for Data Manipulation')
        df_meta = pd.DataFrame({'column': df.columns,
                                'missing_perc': df.isna().sum() * percentage,
                                'unique_perc': df.nunique() * percentage,
                                'dtype': df.dtypes})

        print(df_meta[['missing_perc', 'unique_perc', 'dtype']].round(2))

        print(
            f'ALERT: Droping columns with missing values > {missing_values_acceptable}% :')
        print(df_meta[df_meta['missing_perc'] > missing_values_acceptable]['missing_perc'])
        df_meta = df_meta[df_meta['missing_perc'] <= missing_values_acceptable]

        print(
            f'ALERT: Droping columns with unique values >= {unique_values_acceptable}% :')
        print(df_meta[df_meta['unique_perc'] >= unique_values_acceptable]['unique_perc'])
        df_meta = df_meta[df_meta['unique_perc'] < unique_values_acceptable]

        print('Creating list with numeric features')
        self.numeric_features = list(df_meta[(df_meta['dtype'] == 'int64') | (
            df_meta['dtype'] == 'float')]['column'])
        if self.data.name_target in self.numeric_features:
            self.numeric_features.remove(self.data.name_target)
        print(f'Numeric Feature >>>> {self.numeric_features}')

        print('Creating list with categoric features')
        self.categoric_features = list(
            df_meta[(df_meta['dtype'] == 'object')]['column'])
        print(f'Categoric Feature >>>> {self.categoric_features}')

        print('Feature Normalization and Encoding:')
        if is_train_stage:
            print('Setting Y as target and Removing target from train dataframe')
            y = df[self.data.name_target].fillna(0)
            df = df.drop(columns={self.data.name_target})

            self.feature_names = self.numeric_features + self.categoric_features
            self.scaler = StandardScaler()
            self.catb = ce.CatBoostEncoder(cols=self.categoric_features)

            print('Feature Fit and Transform in train dataframe')
            df[self.numeric_features] = self.scaler.fit_transform(
                df[self.numeric_features])
            df[self.categoric_features] = self.catb.fit_transform(
                df[self.categoric_features], y=y)
            # TODO implementar testes automatizados para garantir que os dados de x e y continuam correspondentes
            return df[self.feature_names], y
        else:
            print('Feature Transform in test dataframe')
            df[self.numeric_features] = self.scaler.transform(
                df[self.numeric_features])
            df[self.categoric_features] = self.catb.transform(
                df[self.categoric_features])
            for column in df[self.feature_names].columns:
                # TODO imputar com a média de teste que deve ser armazenada em algum local
                df[column] = df[column].fillna(df[column].mean())
            return df[self.feature_names]
