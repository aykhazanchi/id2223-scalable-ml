import modal

LOCAL = False


def g():
    import pandas as pd
    import hopsworks
    import joblib
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import requests

    dead_url = "https://raw.githubusercontent.com/aykhazanchi/id2223-scalable-ml/master/lab1/titanic/assets/0.jpg"
    alive_url = "https://raw.githubusercontent.com/aykhazanchi/id2223-scalable-ml/master/lab1/titanic/assets/1.jpg"

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")

    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    offset = 1
    passenger = y_pred[y_pred.size-offset]
    passenger_url = dead_url if passenger == 0 else alive_url

    print("Passenger predicted: " + ("dead" if passenger == 0 else "alive"))
    img = Image.open(requests.get(passenger_url, stream=True).raw)
    img.save("./latest_titanic.png")

    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_titanic.png", "Resources/images", overwrite=True)

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()
    #print(df)
    label = df.iloc[-offset]["survived"]
    label_url = dead_url if label == 0 else alive_url
    print("Passenger actual: " + ("dead" if label == 0 else "alive"))
    img = Image.open(requests.get(label_url, stream=True).raw)
    img.save("./actual_titanic.png")
    dataset_api.upload("./actual_titanic.png", "Resources/images", overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic survival Prediction/Outcome Monitoring")

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [passenger],
        'label': [label],
        'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_titanic_recent.png', table_conversion='matplotlib')
    dataset_api.upload("./df_titanic_recent.png", "Resources/images", overwrite=True)

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our titanic_predictions feature group has
    # examples of all states (alive/dead)
    print("Number of different survival predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)

        df_cm = pd.DataFrame(results, ['True Alive', 'True Dead'],
                             ['Pred Alive', 'Pred Dead'])

        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./titanic_confusion_matrix.png")
        dataset_api.upload("./titanic_confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different survival predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different survival predictions")


if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install([
        "hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn", "xgboost", "dataframe-image"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f_batch():
        g()


if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        stub.deploy("f_batch")