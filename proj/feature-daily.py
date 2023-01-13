import modal

LOCAL = False

def feature_elec():
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    import datetime
    import requests
    import hopsworks
    import os
    from dotenv import load_dotenv

    #%load_ext dotenv

    #%dotenv -vo .env

    # Get date two days ago (Demand and demand forecast are 2 days behind)
    prediction_date = datetime.datetime.today() - datetime.timedelta(days=2)
    prediction_date = prediction_date.date()
    print(prediction_date)

    url = ('https://api.eia.gov/v2/electricity/rto/daily-region-data/data/'
        '?frequency=daily'
        '&data[0]=value'
        '&facets[respondent][]=NY'
        '&facets[timezone][]=Eastern'
        '&facets[type][]=D'
        '&facets[type][]=DF'
        '&sort[0][column]=period'
        '&sort[0][direction]=desc'
        '&offset=0'
        '&length=5000')

    url = url + '&start={}&end={}&api_key={}'.format(prediction_date, prediction_date, os.environ.get('EIA_API_KEY'))

    data = requests.get(url).json()['response']['data']

    # To be used in inference
    data_forecast = data[0]
    data_forecast = pd.DataFrame(data_forecast, index=[0])
    data_forecast.head()

    data_demand = data[1]
    data_demand = pd.DataFrame(data_demand, index=[0])
    data_demand.head()

    # TODO: we don't need this here (used to compare predictions, UI only?)
    data_forecast = data_forecast[['period', 'value']].rename(columns={'period': 'date', 'value': 'forecast'})
    data_forecast['date'] = pd.to_datetime(data_forecast['date'], infer_datetime_format=True)
    #display(data_forecast.head(5))
    print(data_forecast.head(5))

    # Clean DF to same format as fg
    data_demand = data_demand[['period', 'value']].rename(columns={'period': 'date', 'value': 'demand'})
    data_demand['date'] = pd.to_datetime(data_demand['date'], infer_datetime_format=True)
    #display(data_demand.head())
    print(data_demand.head())

    # Get temperature
    weather_api_key = os.environ.get('WEATHER_API_KEY')
    weather_url = ('http://api.weatherapi.com/v1/history.json'
                '?key={}'
                '&q=New%20York,%20USA'
                '&dt={}').format(weather_api_key, prediction_date)

    weather_data = requests.get(weather_url).json()['forecast']['forecastday'][0]['day']['avgtemp_c']
    print(weather_data)
    weather_df = pd.DataFrame({'date': [prediction_date], 'temperature': [weather_data]})
    weather_df['date'] = pd.to_datetime(weather_df['date'], infer_datetime_format=True)
    print(weather_df)

    merged_df = pd.merge(weather_df,data_demand,how='inner', on='date')
    merged_df['day'] = merged_df['date'].dt.dayofweek
    merged_df['month'] = merged_df['date'].dt.month
    merged_df.head()

    # Get bank holidays
    holidays = calendar().holidays(start=merged_df['date'].min(), end=merged_df['date'].max())
    merged_df['holiday'] = merged_df['date'].isin(holidays).astype(int)
    #display(merged_df.head())
    print(merged_df.head())

    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="ny_elec", version=1)
    fg.insert(merged_df, write_options={"wait_for_job": False})

if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install([
        "hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn", "xgboost", "dataframe-image", "pandas", "datetime", "requests", "os"])

    #@stub.function(image=image, schedule=modal.Period(hours=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    @stub.function(image=image, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def modal_feature_elec():
        feature_elec()

if __name__ == "__main__":
    if LOCAL:
        feature_elec()
    else:
        stub.run("modal_feature_elec")
