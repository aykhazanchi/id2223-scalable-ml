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


def predict():
    today = get_date()
    temp = get_temp(today)
    df = pd.DataFrame({"date": [today], "temperature": [temp]})
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df['day'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    holidays = calendar().holidays(start=df['date'].min(), end=df['date'].max())
    df['holiday'] = df['date'].isin(holidays).astype(int)

    demand = model.predict(df.drop(columns=['date']))[0]
    return [today, temp, demand]


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
    description ="Daily NY Electricity Demand Prediction",
    allow_flagging = "never",
    inputs = [],
    outputs = [
        gr.Textbox(label="Date"),
        gr.Textbox(label="Temperature forecast [℃]"),
        gr.Textbox(label="Predicted demand [MWh]"),
    ]
)

# TODO: we have only the demand predictions for two days ago, so we have two options
#  - skip EIA demand forecast (no comparison)
#  - show prediction for two days ago

# TODO: allow custom date/temp input (default to today)?

# TODO: have done some versioning mess (see reqs file)

demo.launch()
