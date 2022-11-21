import os
import modal

LOCAL = False


def g():
    import hopsworks
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import joblib
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema

    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login()
    fs = project.get_feature_store()

    # The feature view is the input set of features for your model. The features can come from different feature groups.
    # You can select features from different feature groups and join them together to create a feature view
    try:
        feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    except:
        titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
        query = titanic_fg.select_except(["passengerid"]) # everything was sanitized to lowercase!
        feature_view = fs.create_feature_view(name="titanic_modal",
                                              version=1,
                                              description="Read from Titanic dataset",
                                              labels=["survived"],
                                              query=query)

    # You can read training data, randomly split into train/test sets of features (X) and labels (y)
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())

    # Evaluate model performance using the features from the test set (X_test)
    y_pred = model.predict(X_test)

    # Compare predictions (y_pred) with the labels in the test setx (y_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    results = confusion_matrix(y_test, y_pred)

    # Create the confusion matrix as a figure, we will later store it as a PNG image file
    df_cm = pd.DataFrame(results, ["True Survived", "True Died"],
                         ["Pred Survived", "Pred Died"])
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()

    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()

    # The contents of the 'titanic_model' directory will be saved to the model registry. Create the dir, first.
    model_dir = "titanic_model"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/titanic_model.pkl")
    fig.savefig(model_dir + "/confusion_matrix.png")

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    titanic_model = mr.python.create_model(
        name="titanic_modal",
        metrics={"accuracy": metrics["accuracy"]},
        model_schema=model_schema,
        description="Titanic Survival Predictor"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    titanic_model.save(model_dir)


if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install([
        "hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn", "xgboost"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        with stub.run():
            f()
            