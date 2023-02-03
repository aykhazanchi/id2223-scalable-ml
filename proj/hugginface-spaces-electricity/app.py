import gradio as gr
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import hopsworks
import joblib
import datetime
import os
import requests

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("ny_elec_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/ny_elec_model.pkl")


def predict(today, temp, holiday):
    if str(today) == '':
        today = get_date()
    if str(temp) == '':
        temp = get_temp(today)
    
    df = pd.DataFrame({"date": [today], "temperature": [temp]})
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df['day'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    if holiday == 0 or holiday == 1:
        df['holiday'] = int(holiday)
    else:
        holidays = calendar().holidays(start=df['date'].min(), end=df['date'].max())
        df['holiday'] = df['date'].isin(holidays).astype(int)
        
    demand = model.predict(df.drop(columns=['date']))[0]

    holiday_label = 'default'
    if holiday == 0:
        holiday_label = 'User selection: working day'
    elif holiday == 1:
        holiday_label = 'User selection: bank holiday'
    elif holiday == 2:
        if int(df['holiday']) == 0:
            holiday_label = 'US Federal Holiday calendar: working day'
        else:
            holiday_label = 'US Federal Holiday calendar: bank holiday'

    return [today, temp, holiday_label, demand]


def get_date():
    today = datetime.datetime.today()
    return today.date()


def get_temp(date):
    weather_api_key = os.environ.get('WEATHER_API_KEY')
    weather_url = ('http://api.weatherapi.com/v1/history.json'
                   '?key={}'
                   '&q=New%20York,%20USA'
                   '&dt={}').format(weather_api_key, date)
    return requests.get(weather_url).json()['forecast']['forecastday'][0]['day']['avgtemp_c']


demo = gr.Interface(
    fn = predict,
    title = "NY Electricity Demand Prediction",
    description ="Daily NY electricity demand prediction, based on current date (day, holiday, ...) and (forecast) temperature average.",
    allow_flagging = "never",
    inputs = [
        gr.Textbox(label="Date"),
        gr.Textbox(label="Temperature forecast [℃]"),
        gr.Radio(choices=["Working day", "Bank holiday", "Check against calendar"], value="Check against calendar", type="index", label="Type of day"),
    ],
    outputs = [
        gr.Textbox(label="Date"),
        gr.Textbox(label="Temperature forecast [℃]"),
        gr.Textbox(label="Type of day"),
        gr.Textbox(label="Predicted demand [MWh]"),
    ]
)

# TODO: we have only the demand predictions for two days ago, so we have two options
#  - skip EIA demand forecast (no comparison)
#  - show prediction for two days ago

# TODO: allow custom date/temp input (default to today)?

# TODO: have done some versioning mess (see reqs file)

demo.launch()
