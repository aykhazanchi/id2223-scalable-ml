import modal

LOCAL = False

def batch_elec():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    from datetime import datetime
    from sklearn.metrics import mean_absolute_error
    import dataframe_image as dfi
    import requests
    import hopsworks
    import joblib
    import os
    from dotenv import load_dotenv
    #%load_ext dotenv

    #%dotenv -vo .env

    project = hopsworks.login()
    fs = project.get_feature_store()

    # get feature group
    # TODO: we could also just read the feature view but that doesn't include the
    #  date of the latest entry. Shouldn't be a problem if both run on the same day
    #  but just to be sure we'll get the complete entry from the group (see below).
    feature_group = fs.get_feature_group(name="ny_elec", version=1)
    #display(feature_group.show(5))
    print(feature_group.show(5))


    # model
    mr = project.get_model_registry()
    model = mr.get_model("ny_elec_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/ny_elec_model.pkl")

    offset = 1
    X_pred = feature_group.read().tail(offset)
    #display(X_pred)
    print(X_pred)


    # predict and get latest (daily) feature
    y_pred = model.predict(X_pred.drop(columns=['demand', 'date']))
    #display(y_pred)
    print(y_pred)


    prediction_date = X_pred.iloc[0]['date']
    prediction_date = prediction_date.date()
    #display(prediction_date)
    print(prediction_date)


    # get demand (forecast)
    url = ('https://api.eia.gov/v2/electricity/rto/daily-region-data/data/'
        '?frequency=daily'
        '&data[0]=value'
        '&facets[respondent][]=NY'
        '&facets[timezone][]=Eastern'
        '&facets[type][]=DF'
        '&sort[0][column]=period'
        '&sort[0][direction]=desc'
        '&offset=0'
        '&length=5000')

    url = url + '&start={}&end={}&api_key={}'.format(prediction_date, prediction_date, os.environ.get('EIA_API_KEY'))

    data = requests.get(url).json()['response']['data']

    #display(data)
    print(data)


    forecast = data[0]['value']
    #display(forecast)
    print(forecast)

    #display(X_pred.iloc[0]['demand'])
    print(X_pred.iloc[0]['demand'])

    #display(y_pred[0])
    print(y_pred[0])


    # DF for monitoring data
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': y_pred,
        'actual': [X_pred.iloc[0]['demand']],
        'forecast_eia': [forecast],
        'prediction_date': [prediction_date],
        'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    #display(monitor_df)
    print(monitor_df)


    # create monitoring FG
    monitor_fg = fs.get_or_create_feature_group(name="ny_elec_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="NY Electricity Prediction/Outcome Monitoring")

    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    # TODO: commented for now since we can wait in a notebook, remember to uncomment
    #  if running e.g. in a modal job!
    #history_df = pd.concat([history_df, monitor_df])
    #display(history_df)
    print(history_df)


    # MAE
    y_pred = history_df['prediction']
    y_test = history_df['actual']
    mean_error = mean_absolute_error(y_test, y_pred)
    #display(mean_error) # in MWh
    print(mean_error) # in MWh

    # TODO: compute "live" in UI

    # create "recents" table for UI and upload
    dataset_api = project.get_dataset_api()
    dfi.export(history_df.tail(5), './df_ny_elec_recent.png', table_conversion='matplotlib')
    dataset_api.upload("./df_ny_elec_recent.png", "Resources/images", overwrite=True)

    # create "prediction" chart for UI and upload
    data = {'label': ['Predicted demand', 'Actual demand', 'EIA forecast'],
            'value': [monitor_df[l][0] for l in ['prediction', 'actual', 'forecast_eia']]}
    pred_df = pd.DataFrame(data)
    pred_plot = sns.barplot(data=pred_df, y='value', x='label')
    plt.ylabel('Demand [MWh]')
    plt.xlabel('')
    plt.ylim(pred_df['value'].min() - 10000, pred_df['value'].max() + 5000)
    plt.title('Predicted and actual demands for {}'.format(monitor_df['prediction_date'][0]))
    fig = pred_plot.get_figure()
    fig.savefig("./df_ny_elec_prediction.png")
    dataset_api.upload("./df_ny_elec_prediction.png", "Resources/images", overwrite=True)

    # create MAE trend graph for UI and upload
    latest_history_df = history_df.loc[-5:] # TODO: might want/need to change this somewhen
    #display(latest_history_df)
    print(latest_history_df)

    no_entries = len(latest_history_df)
    mae = []
    for i in range(no_entries):
        df = latest_history_df.loc[:i]
        mae.append([mean_absolute_error(df['actual'], df['prediction']),
                    mean_absolute_error(df['actual'], df['forecast_eia']),
                    pd.to_datetime(df['datetime'][i]).date()])
    mae_df = pd.DataFrame(mae, columns=['Prediction', 'EIA forecast', 'Date'])
    #display(mae_df)
    print(mae_df)


    mae_plot = sns.lineplot(data=mae_df.melt(id_vars=['Date'],
                                            value_vars=['Prediction', 'EIA forecast']),
                            x='Date', y='value', hue='variable')
    plt.ylabel('Demand [MWh]')
    plt.title('Mean absolute error (MAE) for last {} predictions'.format(no_entries))
    mae_plot.legend().set_title('MAE')
    fig = mae_plot.get_figure()
    fig.savefig("./df_ny_elec_mae.png")
    dataset_api.upload("./df_ny_elec_mae.png", "Resources/images", overwrite=True)

if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install([
        "hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn", "xgboost", "dataframe-image", "matplotlib", "numpy", "pandas", "datetime", "requests", "os"])
    
    #@stub.function(image=image, schedule=modal.Period(hours=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    @stub.function(image=image, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def modal_batch_elec():
        batch_elec()

if __name__ == "__main__":
    if LOCAL:
        batch_elec()
    else:
        stub.run("modal_batch_elec")
