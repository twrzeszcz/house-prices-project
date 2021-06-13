import pandas as pd
import streamlit as st
import numpy as np
import pickle
import cv2
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from utils import all_purpose_transformer, custom_imputer, skewness_remover

st.set_option('deprecation.showPyplotGlobalUse', False)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


import warnings


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                          "provide get_feature_names. "
                          "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [f for f in column]

        return [f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names

def main_section():
    st.title('House Prices Prediction Project')
    background_im = cv2.imread('streamlit/images/background.jpeg')
    st.image(cv2.cvtColor(background_im, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.markdown('**Data Analysis** section contains some basic information about the test and train data and allows to perform EDA '
                'with multiple visualization options. In the **Models Performance** section comparison of different models is presented '
                'with cross-validations scores and train scores. **Feature Importances** section contains calculated feature importances. '
                'Finally **Prediction Service** section allows to choose a record from a test dataset for inference and predict a house price '
                'with the best model.')
    del background_im
    gc.collect()

@st.cache
def load_preprocessors():
    prep_pipe = pickle.load(open('prep_pipe.pkl', 'rb'))
    one_hot_enc = pickle.load(open('one_hot_enc.pkl', 'rb'))
    scaler = pickle.load(open('scale.pkl', 'rb'))

    return prep_pipe, one_hot_enc, scaler

def preprocess_general(df):
    prep_pipe, one_hot_enc, scaler = load_preprocessors()
    return scaler.transform(one_hot_enc.transform(prep_pipe.transform(df)).toarray())

def preprocess_train(df_train):
    df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index, inplace=True)
    y = np.log1p(df_train['SalePrice'])
    df_train.drop('SalePrice', axis=1, inplace=True)

    return df_train, y

@st.cache
def load_data(df_type):
    if df_type == 'Train':
        df_train = pd.read_csv('streamlit/train.csv')
        return df_train

    elif df_type == 'Test':
        df_test = pd.read_csv('streamlit/test.csv')
        return df_test

    elif df_type == 'Full':
        df_train = pd.read_csv('streamlit/train.csv')
        df_test = pd.read_csv('streamlit/test.csv')
        df_full = pd.concat([df_train, df_test]).reset_index(drop=True)
        del df_test, df_train
        return df_full

@st.cache
def train():
    df_train = pd.read_csv('streamlit/train.csv')
    df_train, y = preprocess_train(df_train)
    X_train_prep = preprocess_general(df_train)
    lgbm_model = pickle.load(open('streamlit/models/lgb_model.pkl', 'rb'))
    xgbm_model = pickle.load(open('streamlit/models/xgb_model.pkl', 'rb'))
    models_cross_val = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'KNeighbors Regressor': KNeighborsRegressor(n_jobs=-1),
        'Support Vector Regressor': SVR(),
        'Ridge tuned': Ridge(alpha=18.0, solver='svd'),
        'Lasso tuned': Lasso(alpha=0.0005040816326530613),
        'ElasticNet tuned': ElasticNet(alpha=0.0009081632653061226),
        'Support Vector Regressor tuned': SVR(C=0.014444444444444444, kernel='linear'),
    }

    models_to_train = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_jobs=-1),
        'KNeighbors Regressor': KNeighborsRegressor(n_jobs=-1),
        'Support Vector Regressor': SVR(),
        'XGBoost Regressor': XGBRegressor(n_jobs=-1),
        'CatBoost Regressor': CatBoostRegressor(),
        'LGBM Regressor': LGBMRegressor(),
        'ExtraTrees Regressor': ExtraTreesRegressor(n_jobs=-1),
        'Ridge tuned': Ridge(alpha=18.0, solver='svd'),
        'Lasso tuned': Lasso(alpha=0.0005040816326530613),
        'ElasticNet tuned': ElasticNet(alpha=0.0009081632653061226),
        'Support Vector Regressor tuned': SVR(C=0.014444444444444444, kernel='linear'),
    }

    trained_models_cross_val_scores = {}
    trained_models_scores = {}
    trained_models = {}

    trained_models_cross_val_scores['Random Forest Regressor'] = np.round(0.13740521398712335, 4)
    trained_models_cross_val_scores['XGBoost Regressor'] = np.round(0.1292099819166186, 4)
    trained_models_cross_val_scores['CatBoost Regressor'] = np.round(0.11208948975490034, 4)
    trained_models_cross_val_scores['LGBM Regressor'] = np.round(0.12874397419320133, 4)
    trained_models_cross_val_scores['ExtraTrees Regressor'] = np.round(0.1308816855267648, 4)
    trained_models_cross_val_scores['XGBoost Regressor tuned'] = np.round(0.11346169284890685, 4)
    trained_models_cross_val_scores['LGBM Regressor tuned'] = np.round(0.11824393368870169, 4)
    trained_models['LGBM Regressor tuned'] = lgbm_model.best_estimator_
    trained_models['XGBoost Regressor tuned'] = xgbm_model.best_estimator_
    trained_models['Stacking Regressor'] = pickle.load(open('stacking_model.pkl', 'rb'))
    trained_models['Averaging Regressor'] = pickle.load(open('streamlit/models/avg_model.pkl', 'rb'))


    trained_models_cross_val_scores['Averaging Regressor'] = np.round(0.10829346967266587, 4)
    trained_models_cross_val_scores['Stacking Regressor'] = np.round(0.10749569312383064, 4)

    for name, algo in models_cross_val.items():
        score = np.round(np.sqrt(-cross_val_score(algo, X_train_prep, y,
                                         scoring='neg_mean_squared_error', cv=5, n_jobs=-1)).mean(), 4)
        trained_models_cross_val_scores[name] = score

    for name, algo in models_to_train.items():
        trained_models[name] = algo.fit(X_train_prep, y)
    for name, algo in trained_models.items():
        trained_models_scores[name] = np.sqrt(mean_squared_error(y, algo.predict(X_train_prep)))

    cross_val_scores = pd.DataFrame(list(trained_models_cross_val_scores.items()), columns=['Model', 'RMSLE']).set_index('Model').sort_values(by='RMSLE')
    train_scores = pd.DataFrame(list(trained_models_scores.items()), columns=['Model', 'RMSLE']).set_index('Model').sort_values(by='RMSLE')

    del models_to_train, models_cross_val, trained_models, df_train, X_train_prep, y, lgbm_model, xgbm_model
    gc.collect()
    return cross_val_scores, train_scores

def data_analysis():
    st.title('Data Analysis')
    df_type = st.sidebar.selectbox('Choose dataset', ['Train', 'Test', 'Full'])
    if st.sidebar.checkbox('Load dataset'):
        df = load_data(df_type)
        st.success('Data successfully loaded')
    if st.sidebar.checkbox('Show data'):
        st.dataframe(df)
    if st.sidebar.checkbox('Display shape'):
        st.write('Size of the data: ', df.shape)
    if st.sidebar.checkbox('Display data types'):
        st.write('Data types', df.dtypes)
    if st.sidebar.checkbox('Display missing values'):
        st.write('Missing values', df.isna().sum())
    if st.sidebar.checkbox('Display duplicated rows'):
        st.write('Number of duplicates: ', df.duplicated().sum())
    if st.sidebar.checkbox('Display unique values'):
        st.write('Unique values for each feature: ', df.nunique())
        selected_columns_unique = st.sidebar.selectbox('Select features', df.columns)
        st.write('Unique values for {}'.format(selected_columns_unique), df[selected_columns_unique].unique())
    if st.sidebar.checkbox('Display correlations heatmap'):
        fig, ax = plt.subplots(figsize=(30,20))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)
    if df_type in ['Train', 'Full']:
        if st.sidebar.checkbox('Display correlations between features and target'):
            target_corr = df.corr()['SalePrice'].sort_values(ascending=False)
            target_corr_df = pd.DataFrame({'Correlation': target_corr.values}, index=target_corr.index)
            fig, ax = plt.subplots()
            target_corr_df.plot(kind='bar', figsize=(15, 10), ax=ax)
            st.pyplot(fig)
            st.dataframe(target_corr_df)
    if st.sidebar.checkbox('Display distributions and boxplots'):
        selected_columns_for_distributions_boxplots = st.sidebar.multiselect('Select your preferred numerical features',
                                                                     df.describe().columns)
        for i in selected_columns_for_distributions_boxplots:
            fig, ax = plt.subplots()
            sns.displot(df[i], kde=True)
            st.pyplot()

            fig_1, ax_1 = plt.subplots()
            sns.boxplot(x=df[i])
            st.pyplot()
    if st.sidebar.checkbox('Display countplot'):
        selected_columns_for_countplot = st.sidebar.multiselect('Select your preferred categorical features',
                                                        df.select_dtypes('object').columns)
        plt.rc('legend', fontsize='medium')
        fig, ax = plt.subplots(figsize=(15,10))
        sns.countplot(data=df, y=selected_columns_for_countplot[0], hue=selected_columns_for_countplot[1], ax=ax)
        st.pyplot()
        ax.legend(loc='best')

    if st.sidebar.checkbox('Display barplot'):
        selected_columns_for_barplot = st.sidebar.multiselect('Select 2 categorical features',
                                                                     list(df.select_dtypes('object').columns))
        selected_column_numerical = st.sidebar.selectbox('Select numerical feature', df.describe().columns)
        fig, ax = plt.subplots(figsize=(15,10))
        sns.barplot(data=df, y=selected_columns_for_barplot[0], x=selected_column_numerical, hue=selected_columns_for_barplot[1])
        st.pyplot()
    if st.sidebar.checkbox('Display scatterplot'):
        selected_columns_for_scatterplot = st.sidebar.multiselect('Select 3 numerical features',
                                                                     df.describe().columns)
        selected_column_categorical = st.sidebar.selectbox('Select categorical feature', df.select_dtypes('object').columns)
        fig, ax = plt.subplots(figsize=(15,10))
        sns.scatterplot(data=df, x=selected_columns_for_scatterplot[0], y=selected_columns_for_scatterplot[1],
                    hue=selected_column_categorical, size=selected_columns_for_scatterplot[2])
        st.pyplot()

def model_selection_and_performance():
    cross_val_scores, train_scores = train()
    fig, ax = plt.subplots(figsize=(10,5))
    plt.title('Cross-validation scores')
    chart = sns.barplot(x=cross_val_scores.index, y='RMSLE', data=cross_val_scores)
    for p in chart.patches:
        chart.annotate(format(p.get_height(), '.4f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       size=8,
                       xytext=(0, 5),
                       textcoords='offset points')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    st.pyplot()
    st.write('Cross-validation scores')
    st.dataframe(cross_val_scores)

    fig, ax = plt.subplots(figsize=(10,5))
    plt.title('Train scores')
    chart = sns.barplot(x=train_scores.index, y='RMSLE', data=train_scores)
    for p in chart.patches:
        chart.annotate(format(p.get_height(), '.4f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       size=8,
                       xytext=(0, 5),
                       textcoords='offset points')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    st.pyplot()
    st.write('Train scores')
    st.dataframe(train_scores)

def feature_importances():
    st.title('Feature Importances')
    st.markdown('Feature importances were calculated on the preprocessed data!')
    cat_model = pickle.load(open('streamlit/models/cat_model.pkl', 'rb'))
    df_train = pd.read_csv('streamlit/train.csv')
    df_train, y = preprocess_train(df_train)
    prep_pipe, _, _ = load_preprocessors()
    X_train = prep_pipe.transform(df_train)
    num_cols = X_train.describe().columns

    num_pipe = Pipeline([
        ('scale', RobustScaler())
    ])

    col_pipe = ColumnTransformer([
        ('num pipe', num_pipe, num_cols),
    ], remainder='passthrough')

    X_train_prep = col_pipe.fit_transform(X_train)
    X_train_prep_df = pd.DataFrame(X_train_prep, columns=get_feature_names(col_pipe))
    importances_cat = pd.DataFrame({'Importances': cat_model.feature_importances_}, index=get_feature_names(col_pipe))
    col_number = st.sidebar.slider('Number of features to display', 1, len(X_train_prep_df.columns))
    fig, ax = plt.subplots()
    importances_cat['Importances'].sort_values(ascending=False)[:col_number].plot(kind='bar', figsize=(15, 10), ax=ax)
    st.pyplot()


    st.write('Feature Importances')
    st.dataframe(importances_cat['Importances'].sort_values(ascending=False))

    del cat_model, df_train, X_train, X_train_prep, X_train_prep_df

def prediction_service():
    df_test = pd.read_csv('streamlit/test.csv')
    model = pickle.load(open('stacking_model.pkl', 'rb'))
    selected_record = st.sidebar.selectbox('Select record for prediction', df_test.index)
    st.write('Selected record')
    st.dataframe(df_test.loc[selected_record])
    record_prep = preprocess_general(pd.DataFrame([df_test.loc[selected_record]]))
    st.write('Predicted house price:', np.round(np.expm1(model.predict(record_prep)), 1))

    del df_test, model


activities = ['Main', 'Data Analysis', 'Models Performance', 'Feature Importances', 'Prediction Service', 'About']
option = st.sidebar.selectbox('Select Option', activities)

if option == 'Main':
    main_section()

if option == 'Data Analysis':
    data_analysis()
    gc.collect()

if option == 'Models Performance':
    model_selection_and_performance()
    gc.collect()

if option == 'Feature Importances':
    feature_importances()
    gc.collect()

if option == 'Prediction Service':
    prediction_service()
    gc.collect()

if option == 'About':
    st.title('About')
    st.write('This is an interactive website for the House Prices Prediction Project. Data was taken from kaggle competition.')