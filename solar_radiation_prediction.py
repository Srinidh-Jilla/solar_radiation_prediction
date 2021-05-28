import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
import base64

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

# Background
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1463173904305-ba479d2123b7?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1652&q=80");
        background-size: 180% 180%;
	background-repeat: no-repeat;
        background-position: center;
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Solar Radiation Prediction')

# Body
st.write("""
These datasets are meteorological data from the HI-SEAS weather station from four months (September through December 2016) between Mission IV and Mission V.

In order to predict the solar irradiance, the application will run the chosen parameters through a number of algorithms based on 32.686 rows of data.

The intercorrelation heatmap will allow you to understand the impact between parameters.   
After choosing your parameters, click on "Predict" and wait until the result is displayed.  
Then you will be able to see further details including ML Models, Predicted Values and Raw Data by selecting the optional functions.
   """)

# Loading CSV and dropping irrelevant columns
@st.cache
def load_data():
	df = pd.read_csv('solar_prediction.csv')
	df = df.drop(['UNIXTime', 'Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
	df = df[df['Radiation']>0]
	return df

df = load_data()

if st.checkbox('Show Intercorrelation Heatmap'):
	st.subheader('Intercorrelation Matrix Heatmap')
	df.to_csv('output.csv', index=False)
	df_hm = pd.read_csv('output.csv')

	corr = df_hm.corr()
	mask = np.zeros_like(corr)
	mask[np.triu_indices_from(mask)] = True
	with sns.axes_style("white"):
		f, ax = plt.subplots(figsize=(7,5))
		ax = sns.heatmap(corr, mask=mask, vmax=1)
	st.pyplot(f)

# Sidebar Parameters
st.sidebar.subheader('Input Parameters')
params = {'Temperature' : st.sidebar.slider('Temperature (Fahrenheit)', 34, 71, 51),
          'Pressure' : st.sidebar.slider('Barometric Pressure (Hg)', 30.19, 30.56, 30.42),
	      'Humidity' : st.sidebar.slider('Humidity (percent)', 8, 103, 75),
	      'WindDirection(Degrees)' : st.sidebar.slider('Wind Direction (degrees)', 0.09, 359.95, 143.49),
	      'Speed' : st.sidebar.slider('Speed (mph)', 0.00, 40.50, 6.24)}

# Assigning User Parameters
def user_parameters(df):
	df = df[df['Temperature']==params['Temperature']]
	df = df[df['Pressure']==params['Pressure']]
	df = df[df['Humidity']==params['Humidity']]
	df = df[df['WindDirection(Degrees)']==params['WindDirection(Degrees)']]
	df = df[df['Speed']==params['Speed']]
	df.reset_index()
	return df

test_size = st.sidebar.slider('Pick Test Size', 0.05, 0.5, 0.25, step=0.05)

# Model and ML Algorithms
@st.cache
def get_models():
	y = df['Radiation']
	X = df[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
	models = [DummyRegressor(strategy='mean'),
	          RandomForestRegressor(n_estimators=170, max_depth=25),
	          DecisionTreeRegressor(max_depth=30),
	          GradientBoostingRegressor(learning_rate=0.01, n_estimators=200, max_depth=5),
	          LinearRegression(n_jobs=10, normalize=True)]
	df_models = pd.DataFrame()
	temp = {}
	print(X_test)
	#run through models
	for model in models:
		print(model)
		m = str(model)
		temp['Model'] = m[:m.index('(')]
		model.fit(X_train, y_train)
		temp['RMSE_Radiation'] = sqrt(mse(y_test, model.predict(X_test)))
		temp['Pred Value'] = model.predict(pd.DataFrame(params, index=[0]))[0]
		print('RMSE score', temp['RMSE_Radiation'])
		df_models = df_models.append([temp])
	df_models.set_index('Model', inplace=True)
	pred_value = df_models['Pred Value'].iloc[[df_models['RMSE_Radiation'].argmin()]].values.astype(float)
	return pred_value, df_models

def run_data():
	df_models = get_models()[0][0]
	st.write('Given your parameters, solar radiation would be **{:.2f}** W/m2.'.format(df_models))

def show_ML():
	df_models = get_models()[1]
	df_models
	st.write('**This diagram shows Root Mean Square Error for all models:**')
	st.bar_chart(df_models['RMSE_Radiation'])
	st.write('**Predicted Values:**')
	st.area_chart(df_models['Pred Value'])

btn = st.sidebar.button('Predict')

if btn:
	st.subheader('Multi Model Prediction')
	run_data()
	show_ML()
else:
	pass

# Additional Functions
st.sidebar.subheader('Additional Functions')

if st.sidebar.checkbox('Show Raw Data'):
	df

# Download raw data
def filedownload(df):
	csv = df.to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="solar_radiation.csv">Download Dataset (CSV)</a>'
	return href

st.sidebar.markdown(filedownload(df), unsafe_allow_html=True)
