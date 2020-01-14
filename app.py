import streamlit as st
import pandas as pd
import pickle
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from datetime import datetime, timedelta
import requests
from datetime import date
import pytz
import urllib
import urllib.parse
import json
import plotly.graph_objects as go
from urllib.request import urlopen
from weather import *

def datetime_to_date(my_datetime):
    return my_datetime.date()

def main():
    st.title('JGH ED Visit Predictor')
    model_select = st.sidebar.selectbox('Prediction Model',('2-week daily with weather', '72-hour hourly with weather','1-year daily'))

    my_placeholder = st.empty()
    my_placeholder.text("Loading model...")

    if model_select == '2-week daily with weather':
        model = 'daily'
        pkl_path = "daily-20-01-12.pkl"
    elif model_select == '72-hour hourly with weather':
        model = 'hourly'
        pkl_path = "hourly-20-01-12.pkl"
    elif model_select == '1-year daily':
        model = 'year'
        pkl_path = "longterm-20-01-12.pkl"

    # read the Prophet model object
    with open(pkl_path, 'rb') as f:
        m = pickle.load(f)

    # if model == 'daily':
    #     weather_forecast_on_file = pd.read_csv('Montreal-daily.csv')
    # elif model == 'hourly':
    #     weather_forecast_on_file = pd.read_csv('Montreal-hourly.csv')

    # if model in ['daily','hourly']:
    #     st.write("First weather date on file: "+weather_forecast_on_file.ds.min())

    now = datetime.now().astimezone(pytz.timezone('US/Eastern'))
    mtl_now = now.replace(tzinfo=None)
    local_now = datetime.now()
    today = now.date()
    st.write('Currently: '+str(now))
    #st.write('mtl_now: '+str(mtl_now))
    #st.write('local_now: '+str(local_now))
    #st.write('mtl_now + 72 hours: '+str(mtl_now+timedelta(hours=72)))

    if model in ['daily','hourly']:
        # if str(weather_forecast_on_file.ds.min()) == str(today):
        #     st.write('Using weather forecast already on file')
        #     weather_forecast = weather_forecast_on_file
        # else:
            my_placeholder.text("Fetching the weather forecast...")
            if model == 'daily':
                frequency = 24
                num_of_days = 15
            elif model == 'hourly':
                frequency = 1
                num_of_days = 5
            api_key = '3d51d04f983a478e90f164916191012'
            location_list = ['Montreal']

            retrieve_future_data(api_key,location_list,frequency,num_of_days)

            if model == 'daily':
                weather_forecast = pd.read_csv('Montreal-daily.csv')
            elif model == 'hourly':
                weather_forecast = pd.read_csv('Montreal-hourly.csv')
            weather_forecast['ds'] = pd.to_datetime(weather_forecast['ds'])
            weather_forecast['ds'] = weather_forecast['ds'].apply(lambda x: x.replace(tzinfo=None))
            # weather_forecast['ds'].apply(lambda x: x.tz_localize('GMT'))
            #weather_forecast['ds'].apply(lambda x: x.astimezone(pytz.timezone('US/Eastern')))

    my_placeholder.text("Making predictions...")
    
    if model == 'daily':
        forecast = m.predict(weather_forecast)
    elif model == 'hourly':
        mask = (weather_forecast.ds >= mtl_now) & (weather_forecast.ds <= mtl_now+timedelta(hours=72))
        forecast = m.predict(weather_forecast.loc[mask])
    elif model == 'year':
        future = m.make_future_dataframe(365,'D')
        #mask = (future.ds >= mtl_now)
        #forecast = m.predict(future.loc[mask])
        forecast = m.predict(future)

    st.header('Predictions')

    if model in ['daily','hourly']:
        x = [str(a) for a in forecast.ds.to_list()]
    elif model == 'year':
        predictions = forecast[forecast.ds >= mtl_now]
        x = [str(a) for a in predictions.ds.to_list()]
    y = forecast.yhat.to_list()

    fig = go.Figure(data=[go.Scatter(x=x, y=y)])

    st.plotly_chart(fig)
    #st.write(forecast[['ds', 'yhat']])

    if model in ['daily','hourly']:
        st.header('Weather Forecast')
        if model == 'daily':
            st.write(weather_forecast)
            st.subheader('Factors affecting the prediction:')
            date_picker = st.selectbox('Choose Date',forecast.ds.to_list(),format_func=datetime_to_date)
            forecast.set_index("ds", inplace=True)
            daily_factors = forecast[['holidays','maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour', 'uvIndex', 
               'moon_illumination', 
               'DewPointC',  'FeelsLikeC', 'HeatIndexC', 'WindChillC', 'WindGustKmph',
               'cloudcover', 'humidity', 'precipMM', 'pressure', 'tempC', 'visibility',
               'winddirDegree', 'windspeedKmph']].loc[date_picker].sort_values(ascending=False)
            st.write('Visits predicted: '+str(int(forecast.loc[date_picker]['yhat'].round(0))))
            st.table(daily_factors)
        elif model == 'hourly':
            st.write(weather_forecast.loc[mask])

    if model == 'year':
        fig2 = m.plot_components(forecast)
        st.plotly_chart(fig2)
    
    my_placeholder.text("")


    

if __name__ == "__main__":
    main()