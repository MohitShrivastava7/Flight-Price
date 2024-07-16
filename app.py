import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    PowerTransformer,
    FunctionTransformer,
    StandardScaler
)
from sklearn.pipeline import Pipeline,FeatureUnion
from feature_engine.encoding import (
    RareLabelEncoder,
    MeanEncoder,
    CountFrequencyEncoder,
    OrdinalEncoder
)
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer
import warnings
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import os 
sklearn.set_config(transform_output='pandas')
warnings.filterwarnings('ignore')
train = pd.read_csv("train.csv")
train.head()
x_train = train.drop(columns='price')
y_train = train.price.copy()
# preprocessing operation

air_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('grouper',RareLabelEncoder(tol=0.1,replace_with ='Other',n_categories=2)),
    ('encoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])

air_transformer.fit_transform(x_train.loc[:,['airline']])

feature_to_extract = ['week','month','day_of_week','day_of_year']

doj_transformer = Pipeline(steps=[
    ('dt',DatetimeFeatures(features_to_extract=feature_to_extract,yearfirst=True, format='mixed')),
    ('scaler',MinMaxScaler())
])
doj_transformer.fit_transform(x_train.loc[:,['date_of_journey']])

location_subset = x_train.loc[:,['source','destination']]


location_pipe1=Pipeline(steps=[
    ('grouper',RareLabelEncoder(tol=0.1,replace_with='Other',n_categories=2)),
    ('encoder',MeanEncoder()),
    ('transformer',PowerTransformer())
])
location_pipe1.fit_transform(location_subset,y_train)

def is_north(x):
    columns = x.columns.to_list()
    north_cities=['Delhi','Kolkata','Mumbai','New Delhi']
    return (
        x.
        assign(**{
            f'{col}_is_north':x.loc[:,col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
)
FunctionTransformer(func=is_north).fit_transform(location_subset)

location_transformation = FeatureUnion(transformer_list=[
    ('part1',location_pipe1),
    ('part2',FunctionTransformer(func=is_north))
])
location_transformation.fit_transform(location_subset,y_train)

time_subset = x_train.loc[:,['dep_time','arrival_time']]

time_pipe1 = Pipeline(steps=[
    ('dt',DatetimeFeatures(features_to_extract=['hour','minute'])),
    ('scaler',MinMaxScaler())
])
time_pipe1.fit_transform(time_subset)

def part_of_day(x,morning=4,noon=12,eve=16,night=20):
    columns = x.columns.to_list()
    x_temp = x.assign(**{
        col:pd.to_datetime(x.loc[:,col]).dt.hour
        for col in columns
    })
    return (
        x_temp
        .assign(**{
            f'{col}_part_of_day': np.select(
                [x_temp.loc[:,col].between(morning,noon,inclusive='left'),
                x_temp.loc[:,col].between(noon,eve,inclusive='left'),
                 x_temp.loc[:,col].between(eve,night,inclusive='left')],
                ['morning','afternoon','evening'],
                default = 'night'
            )
            for col in columns
        })
        .drop(columns=columns)
    )
FunctionTransformer(func=part_of_day).fit_transform(time_subset)

time_pipe2 = Pipeline(steps=[
    ('part1',FunctionTransformer(func=part_of_day)),
    ('encoder',CountFrequencyEncoder()),
    ('scaler',MinMaxScaler())
])
time_pipe2.fit_transform(time_subset)

time_transformer =FeatureUnion(transformer_list=[
    ('part1',time_pipe1),
    ('pipe2',time_pipe2)
])
time_transformer.fit_transform(time_subset)

class RBFpercentileSimilarity(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None,percentiles=[0.25,0.50,0.75],gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma
        
    def fit(self,x,y=None):
        if not self.variables:
            self.variables = x.select_dtypes(include='number').columns.to_list()
            
        self.reference_values_ = {
            col:(
                x
                .loc[:,col]
                .quantile(self.percentiles)
                .values
                .reshape(-1,1)
            )
            for col in self.variables
        }
        
        return self
    
    def transform(self,x):
        objects=[]
        for col in self.variables:
            columns = [f'{col}_rbf_int{(percentile*100)}' for percentile in self.percentiles]
            obj = pd.DataFrame(
            data = rbf_kernel(x.loc[:,[col]],self.reference_values_[col],gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
            
        return pd.concat(objects,axis=1)
    
RBFpercentileSimilarity().fit_transform(x_train.loc[:,['duration']])

def duration_category(X, short=180, med=400):
	return (
		X
		.assign(duration_cat=np.select([X.duration.lt(short),
									    X.duration.between(short, med, inclusive="left")],
									   ["short", "medium"],
									   default="long"))
		.drop(columns="duration")
	)

def is_over(X, value=1000):
	return (
		X
		.assign(**{
			f"duration_over_{value}": X.duration.ge(value).astype(int)
		})
		.drop(columns="duration")
	)

duration_pipe1 = Pipeline(steps=[
    # ('rbf',RBFpercentileSimilarity()),
    ('scaler',PowerTransformer())
])

duration_union = FeatureUnion(transformer_list=[
    ('part1',duration_pipe1),
    ('part2',StandardScaler())
])
duration_transformer = Pipeline(steps=[
    ('outlier',Winsorizer(capping_method='iqr',fold=1.5)),
    ('Imputer',SimpleImputer(strategy='median')),
    ('union', duration_union)
])
duration_transformer.fit_transform(x_train.loc[:,['duration']])

(
    x_train
    .duration
    .loc[lambda ser: ser > 1000]
    .to_frame()
    .quantile([0.25,0.50,0.75])
)

def is_direct(X):
	return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))


total_stops_transformer = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="most_frequent")),
	("", FunctionTransformer(func=is_direct))
])

total_stops_transformer.fit_transform(x_train.loc[:, ["total_stops"]])

info_pipe1 = Pipeline(steps=[
	("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
	("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

info_pipe1.fit_transform(x_train.loc[:, ["additional_info"]])

def have_info(X):
	return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))
info_union = FeatureUnion(transformer_list=[
	("part1", info_pipe1),
	("part2", FunctionTransformer(func=have_info))
])
info_transformer = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
	("union", info_union)
])

info_transformer.fit_transform(x_train.loc[:, ["additional_info"]])

column_transformer = ColumnTransformer(transformers=[
    ('air',air_transformer,['airline']),
    ('doj',doj_transformer,['date_of_journey']),
    ('location',location_transformation,['source','destination']),
    ('time',time_transformer,['dep_time','arrival_time']),
    ('dur',duration_transformer,['duration']),
    ("stops", total_stops_transformer, ["total_stops"]),
    ("info", info_transformer, ["additional_info"])
],remainder='passthrough')
column_transformer.fit_transform(x_train,y_train)

estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
	estimator=estimator,
	scoring="r2",
	threshold=0.1
) 

preprocessor = Pipeline(steps=[
	("ct", column_transformer),
	("selector", selector)
])

preprocessor.fit_transform(x_train, y_train)

### fit and save the preprocessor

joblib.dump(preprocessor,'preprocessor.joblib')

### web application
import streamlit as st

st.set_page_config(
     page_title='Flights Prices Prediction',
     page_icon='✈️',
     layout='wide'
)

st.title('Flights Price Prediction')

# User Input
airline = st.selectbox(
     'Airline:',
     options = x_train.airline.unique()
)

# doj
doj = st.date_input('Data of Journey:')

# source
source = st.selectbox(
     'Source:',
     options=x_train.source.unique()
)

# Detination 
destination = st.selectbox(
     'Destination:',
     options=x_train.destination.unique()
)

# dep_time
dep_time = st.time_input('Departure Time:')

# arival_time
arrival_time = st.time_input('Arrival Time:')

# duration
duration = st.number_input(
     'Duration (mins):',
     step=1
)

# total_stops
total_stops = st.number_input(
     'Total Stops:',
     step=1,
     min_value=0
)

# additional_info
additional_info= st.selectbox(
     'Additional Info:',
     options=x_train.additional_info.unique()
)

x_new = pd.DataFrame(dict(
     airline = [airline],
     date_of_journey = [doj],
     source = [source],
     destination = [destination],
     dep_time = [dep_time],
     arrival_time = [arrival_time],
     duration = [duration],
     total_stops = [total_stops],
     additional_info = [additional_info],
)).astype({
     col:'str'
     for col in ['date_of_journey','dep_time','arrival_time']
})

if st.button('Predict'):
    # load preprocessor
    saved_preprocessor = joblib.load('preprocessor.joblib')
    x_new_pre = saved_preprocessor.transform(x_new)

    # load the model
    import pickle
    with open('best_model.pkl','rb') as f:
        model = pickle.load(f)
    x_new_pre = np.array(x_new_pre)
    pred = model.predict(x_new_pre)[0]

    st.info(f'The predicted price is {pred:,.0f}INR')