import category_encoders as ce
import pandas as pd
#from data_source import DataSource
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, data, col_analise=False):
        self.categoric_features = None
        self.numeric_features = None
        self.scaler = None
        self.catb = None
        self.data = data
        self.col_analise = col_analise
        '''
        Class for preprocessing data model.
        :param data: DataSource object
        :param col_analise: List with: col_name : String,
                                        var_type : None, 'cat' or 'num' 
                                        fillna : Int or Float,
                                        encode : Bool,
                                        drop_first : Bool
        :return: Preprocessed object
        '''

    def get_feature_names(self):
        '''
        Get all features names for model 
        :return: All features names (categoric + numeric)
        '''
        return self.numeric_features + self.categoric_features

    def _preprocess_manual(self, df):
        '''
        Manually preprocess dataframe setting categoric and numeric features 
        :param df: Dataframe
        :return: Dataframe processed 
        '''
        name, var_type, fill, encode, drop_first = 0, 1, 2, 3, 4
        self.numeric_features = list()
        self.categoric_features = list()

        for col in self.col_analise:
            if col[var_type]:
                feature = self.categoric_features if col[var_type] == 'cat' else self.numeric_features
                print(f'use: {col[name]}')
                if col[fill] != None:
                    df[col[name]].fillna(col[fill], inplace=True)
                    print(f'\tfill na with: {col[fill]}')
                if col[encode]:
                    values = df[col[name]].value_counts(
                    ).sort_index().index.values
                    df = pd.get_dummies(df,
                                        columns=[col[name]],
                                        drop_first=col[drop_first],
                                        dtype='int')
                    print(f'\tencode: {values}')
                    if col[drop_first]:
                        print(f'\tdrop value: {values[0]}')
                        values = values[1:]
                    feature += [f'{col[name]}_{x}' for x in values]
                else:
                    feature.append(col[name])
            else:
                df.drop(columns=col[name], inplace=True)
                print(f'drop: {col[name]}')
        return df

    def _preprocess_auto(self, df, missing_values_acceptable, unique_values_acceptable):
        '''
        Automatically preprocess dataframe setting categoric and numeric features 
        :param df: Dataframe
        :param missing_values_acceptable: Int with minimum missing values acceptable percentage
        :param unique_values_acceptable: Int with maximum unique values acceptable percentage
        '''
        print('ALERT: Creating DataFrame for Data Manipulation')
        percentage = 100/df.shape[0]
        df_meta = pd.DataFrame({'column': df.columns,
                                'missing_perc': df.isna().sum() * percentage,
                                'unique_perc': df.nunique() * percentage,
                                'dtype': df.dtypes})
        print(df_meta[['missing_perc', 'unique_perc', 'dtype']].round(2))

        print(
            f'ALERT: Droping columns with missing values > {missing_values_acceptable}% :')
        print(df_meta[df_meta['missing_perc'] >
                      missing_values_acceptable]['missing_perc'])
        df_meta = df_meta[df_meta['missing_perc'] <= missing_values_acceptable]

        print(
            f'ALERT: Droping columns with unique values >= {unique_values_acceptable}% :')
        print(df_meta[df_meta['unique_perc'] >=
                      unique_values_acceptable]['unique_perc'])
        df_meta = df_meta[df_meta['unique_perc'] < unique_values_acceptable]

        print('ALERT: Creating list with numeric features:')
        self.numeric_features = list(df_meta[(df_meta['dtype'] == 'int64') | (
            df_meta['dtype'] == 'float')]['column'])
        if self.data.name_target in self.numeric_features:
            self.numeric_features.remove(self.data.name_target)

        print('ALERT: Creating list with categoric features:')
        self.categoric_features = list(
            df_meta[(df_meta['dtype'] == 'object')]['column'])
        if self.data.name_target in self.categoric_features:
            self.categoric_features.remove(self.data.name_target)

    def _process_test(self, df):
        '''
        Process train dataframe with transform
        :param df: Dataframe
        :return: Transformed dataframe 
        '''
        print('Feature Transform in test dataframe')
        df[self.numeric_features] = self.scaler.transform(
            df[self.numeric_features])
        df[self.categoric_features] = self.catb.transform(
            df[self.categoric_features])
        for column in df[self.get_feature_names()].columns:
            # TODO imputar com a média de teste que deve ser armazenada em algum local
            df[column] = df[column].fillna(df[column].mean())
        return df[self.get_feature_names()]

    def _process_train(self, df):
        '''
        Process train dataframe with fit and transform
        :param df: Dataframe
        :return: Fitted and transformed dataframe, and target serie 
        '''
        print('Setting Y as target and Removing target from train dataframe')
        y = df[self.data.name_target].fillna(0)
        df = df.drop(columns={self.data.name_target})

        self.scaler = StandardScaler()
        self.catb = ce.CatBoostEncoder(cols=self.categoric_features)

        print('Feature Fit and Transform in train dataframe')
        df[self.numeric_features] = self.scaler.fit_transform(
            df[self.numeric_features])
        df[self.categoric_features] = self.catb.fit_transform(
            df[self.categoric_features], y=y)

        # TODO implementar testes automatizados para garantir que os dados de x e y continuam correspondentes

        #self.categoric_features = categoric_features
        #self.numeric_features = numeric_features

        return df[self.get_feature_names()], y

    def process(self, is_train_stage=True,
                missing_values_acceptable=0,
                unique_values_acceptable=100):
        '''
        Process data for training the model.
        :param etapa_treino: Boolean
        :param missing_values_acceptable: Int with minimum missing values acceptable percentage
        :param unique_values_acceptable: Int with maximum unique values acceptable percentage
        :return: processed Pandas Data Frame, and target if train stage
        '''
        df = self.data.read_data(is_train_stage)

        if self.col_analise:
            df = self._preprocess_manual(df)
        else:
            self._preprocess_auto(
                df, missing_values_acceptable, unique_values_acceptable)

        # TODO deletar quando o PCA estiver funcionando
        if 'TP_ESCOLA_4' in self.categoric_features:
            self.categoric_features.remove('TP_ESCOLA_4')

        print(f'Numeric Feature >>>> {self.numeric_features}')
        print(f'Categoric Feature >>>> {self.categoric_features}')

        return self._process_train(df) if is_train_stage else self._process_test(df)
