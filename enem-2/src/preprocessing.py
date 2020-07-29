import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self):
        self.feature_names = None
        self.std_scaler = None
        self.categoric_features = None
        self.numeric_features = None
        self.catb = None
        self.scaler = None
        self.train_features = None

    def process(self, df, etapa_treino=True):
        '''
        Process data for training the model.
        :param df: Pandas DataFrame
        :param etapa_treino: Boolean
        :return: processed Pandas Data Frame
        '''
        print('Creating DataFrame for Data Manipulation')
        df_meta = pd.DataFrame({'column': df.columns,
                             'missing_perc': (df.isna().sum() / df.shape[0]) * 100,
                             'dtype': df.dtypes })
        print('Droping columns with missing values')
        df_meta = df_meta[df_meta['missing_perc'] == 0]
        print('Dropping column with Id') # TODO passar como parametro
        df_meta = df_meta[df_meta['column'] != 'Id']
        print('Creating list with numeric features')
        numeric_features = list(df_meta[(df_meta['dtype'] == 'int64') | (df_meta['dtype'] == 'float')]['column'])
        print('Creating list with categoric features')
        categoric_features = list(df_meta[(df_meta['dtype'] == 'object')]['column'])
        
        print('Feature encoder')
        print('Feature Normalization and Encoding')
        std_scaler = StandardScaler()
        if etapa_treino:
            print('Setting Y as target and Removing target')
            y = df['SalePrice']
            numeric_features.remove('SalePrice') # TODO passar como parametro
            df = df.drop(columns={'SalePrice'}) # TODO passar como parametro 
            
            self.numeric_features = numeric_features
            self.categoric_features = categoric_features
            self.feature_names = self.numeric_features + self.categoric_features
            self.scaler = std_scaler
            self.catb = ce.CatBoostEncoder(cols=self.categoric_features)
            df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])
            df[self.categoric_features] = self.catb.fit_transform(df[self.categoric_features], y=y)
            self.train_features = self.numeric_features + self.categoric_features
            return df[self.categoric_features + self.numeric_features], y # TODO implementar testes automatizados para garantir que os dados de x e y continuam correspondentes
        else:
            df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
            df[self.categoric_features] = self.catb.transform(df[self.categoric_features])
            for column in df[self.categoric_features + self.numeric_features].columns:
                df[column] = df[column].fillna(df[column].mean()) # TODO imputar com a média de teste que deve ser armazenada em algum local
            return df[self.categoric_features + self.numeric_features]
