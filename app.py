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



##################################
# function to unnest json for each month
def extract_json_weather_data(data):
    num_days = len(data)
    #print(num_days)
    # initialize df_month to store return data
    df_forecast = pd.DataFrame()
    for i in range(num_days):
        # extract this day
        d = data[i]
        # astronomy data is the same for the whole day
        astr_df = pd.DataFrame(d['astronomy'])
        # hourly data; temperature for each hour of the day
        hourly_df = pd.DataFrame(d['hourly'])
        # this wanted_key will be duplicated and use 'ffill' to fill up the NAs
        wanted_keys = ['date', 'maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour', 'uvIndex'] # The keys you want
        subset_d = dict((k, d[k]) for k in wanted_keys if k in d)
        this_df = pd.DataFrame(subset_d,index=[0])        
        df = pd.concat([this_df.reset_index(drop=True), astr_df], axis=1)
        # concat selected astonomy columns with hourly data
        df = pd.concat([df,hourly_df], axis=1)
        df = df.fillna(method='ffill')
        # make date_time columm to proper format
        # fill leading zero for hours to 4 digits (0000-2400 hr)
        df['time'] = df['time'].apply(lambda x: x.zfill(4))
        # keep only first 2 digit (00-24 hr) 
        df['time'] = df['time'].str[:2]
        # convert to pandas datetime
        df['ds'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        # keep only interested columns
        col_to_keep = ['ds', 'maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour', 'uvIndex', 
               'moon_illumination', 
               'DewPointC',  'FeelsLikeC', 'HeatIndexC', 'WindChillC', 'WindGustKmph',
               'cloudcover', 'humidity', 'precipMM', 'pressure', 'tempC', 'visibility',
               'winddirDegree', 'windspeedKmph']
        df = df[col_to_keep]
        df_forecast = pd.concat([df_forecast,df])
    #return(df_month)
    return(df_forecast)

##################################
#function to retrive data by date range and location
#default frequency = 1 hr
#each month costs 1 request (free trial 500 requests/key, as of 30-May-2019)
def retrieve_weather_data(api_key,location,frequency,num_of_days):
    
    #start_time = datetime.now()
    
    # create list of months, convert to month begins (first day of each month)
    #list_mon_begin= pd.date_range(start_date,end_date, freq='1M')-pd.offsets.MonthBegin(1)
    # convert to Series and append first day of the last month
    #list_mon_begin = pd.concat([pd.Series(list_mon_begin), pd.Series(pd.to_datetime(end_date,infer_datetime_format=True).replace(day=1))], ignore_index=True)
    # change the begin date to start_date
    #list_mon_begin[0] = pd.to_datetime(start_date,infer_datetime_format=True)
    
    # create list of months, convert to month ends (last day of each month)
    #list_mon_end = pd.date_range(start_date,end_date, freq='1M')-pd.offsets.MonthEnd(0)
    # convert to Series and append the end_date
    #list_mon_end = pd.concat([pd.Series(list_mon_end), pd.Series(pd.to_datetime(end_date,infer_datetime_format=True))], ignore_index=True)
    
    # count number of months to be retrieved
    #total_months = len(list_mon_begin)

    # initialize df_hist to store return data
    weather_df = pd.DataFrame()
    #for m in range(total_months):
        
        #start_d =str(list_mon_begin[m])[:10]
        #end_d =str(list_mon_end[m])[:10]
        #print('Currently retrieving data for '+location+': from '+start_d+' to '+end_d)
        
    url_page = 'http://api.worldweatheronline.com/premium/v1/weather.ashx?key='+api_key+'&q='+location+'&format=json&num_of_days='+str(num_of_days)+'&tp='+str(frequency)
    json_page = urllib.request.urlopen(url_page)
    json_data = json.loads(json_page.read().decode())
    data= json_data['data']['weather']
       # call function to extract json object
    weather_df = extract_json_weather_data(data)
    #df_hist = pd.concat([df_hist,df_this_month])
        
    #time_elapsed = datetime.now() - start_time
    #print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    return(weather_df)

##################################
#main function to retrive the data by location list
def retrieve_future_data(api_key,location_list,frequency,num_of_days,location_label = False, export_csv = True, store_df = False):
    result_list = []
    for location in location_list:
        #print('\n\nRetrieving weather data for '+location+'\n\n')
        df_this_city = retrieve_weather_data(api_key,location,frequency,num_of_days)
        
        if (location_label == True):
        # add city name as prefix to the colnames
            df_this_city = df_this_city.add_prefix(location+'_')
            df_this_city.columns.values[0] = 'date_time'    
        
        if (export_csv == True):
            if frequency == 24:
                df_this_city.to_csv('./'+location+'-daily.csv', header=True, index=False) 
            elif frequency == 1:
                df_this_city.to_csv('./'+location+'-hourly.csv', header=True, index=False)
            #print('\n\nexport '+location+' completed!\n\n')
        
        if (store_df == True):
        # save result as object in the work space
            result_list.append(df_this_city)
    return(result_list)
##################################

def datetime_to_date(my_datetime):
    return my_datetime.date()

def main():
    st.title('JGH ED Visit Predictor')
    model = st.sidebar.selectbox('Prediction Model',('2-week daily with weather', '72-hour hourly with weather','1-year daily'))

    my_placeholder = st.empty()
    my_placeholder.text("Loading model...")

    if model == '2-week daily with weather':
        model = 'daily'
        pkl_path = "daily-19-12-22.pkl"
    elif model == '72-hour hourly with weather':
        model = 'hourly'
        pkl_path = "hourly-19-12-22.pkl"
    elif model == '1-year daily':
        model = 'year'
        pkl_path = "longterm-19-12-22.pkl"

    # read the Prophet model object
    with open(pkl_path, 'rb') as f:
        m = pickle.load(f)

    if model == 'daily':
        weather_forecast_on_file = pd.read_csv('Montreal-daily.csv')
    elif model == 'hourly':
        weather_forecast_on_file = pd.read_csv('Montreal-hourly.csv')

    if model in ['daily','hourly']:
        st.write("First weather date on file: "+weather_forecast_on_file.ds.min())

    now = datetime.now().astimezone(pytz.timezone('US/Eastern'))
    mtl_now = now.replace(tzinfo=None)
    local_now = datetime.now()
    today = now.date()
    st.write('Currently: '+str(now))
    #st.write('mtl_now: '+str(mtl_now))
    #st.write('local_now: '+str(local_now))
    #st.write('mtl_now + 72 hours: '+str(mtl_now+timedelta(hours=72)))

    if model in ['daily','hourly']:
        if str(weather_forecast_on_file.ds.min()) == str(today):
            st.write('Using weather forecast already on file')
            weather_forecast = weather_forecast_on_file
        else:
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
            weather_forecast['ds']=pd.to_datetime(weather_forecast['ds'])

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