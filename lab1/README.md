# ID2223 - Lab 1

The goal of the Lab 1 projects was to build and deploy two serverless ML prediction systems using Modal, Hopsworks, and Hugging Face Spaces. The first system works with the Iris flowers dataset and deals with predicting the variety of a synthetically generated Iris Flower. The second system works with the Titanic dataset and predicts the survival chances of synthetically generated passengers. Both systems have an interactive component where a user can submit their own values for a flower/passenger as well as a historical record that gives the model statistics over a batch of data.

## Pipelines
The pipelines for both Iris and Titanic are very closely related. As a result, a brief description is provided of each step in the process which for the most part remains the same between the two datasets, a short description of the data preparation and model choice for the two implementations is given in the respective sections.

**{iris, titanic}-feature-pipeline.py**

Our entire pipeline begins with the feature pipeline which is responsible for reading our datasets, doing feature engineering on them, extracting the schemas that we need for creating our feature groups in Hopsworks, and inserting data into the created feature group. 

**{iris, titanic}-training-pipeline.py**

The model training pipeline runs after the Hopsworks feature groups are created. When run the first time, the model training pipeline creates a feature view from an existing feature group that we created in the feature pipeline. This feature view is the training dataset schema that we will use to train our model. We split this dataset into training data and test data and extract the features and labels into X_train and y_train values used to train and test the models. 

The training pipeline also creates a classification report and a heatmap image of a confusion matrix. The classification report is stored in the Metrics section under the model in Hopsworks and the heatmap is stored as a .png image. The training pipeline also saves the model locally and uploads the same to Hopsworks from where it can be loaded directly for use in Hugging Face.

The training pipeline can be run locally or it can be run on Modal where it can run in a serverless manner.

**{iris, titanic}-feature-pipeline-daily.py**

The feature-daily pipeline randomly generates a synthetic flower/passenger for which we run a prediction using our pre-trained model. The values are generated at random and added to the feature group in Hopsworks. 

The values for our features are generated at random. For the iris dataset, we have a broad range of values for the labels from which we know what variety to expect. Depending on the feature values generated, we can identify the value of the label. As a result, for the Iris dataset, we can use this “true variety” to compare the accuracy of our model against its “predicted variety”. 

For the titanic dataset we also generate random values for the features. However, the survival of a passenger that is completely synthetic cannot be easily declared hence we go with a random label value. The model accuracy is compared to this random label for the titanic dataset. We chose to consider more nuanced approaches to identifying if a particular synthetic passenger would survive or not but this requires creating our own rules which eventually implies including a certain bias into the model as a whole. We do not feel that there is any good way to accurately ascertain the label for a synthetic passenger without introducing some form of bias into the model. Thus for the titanic dataset we chose to go with the random label generation implementation that we feel introduces the least amount of bias.

The feature-pipeline-daily can run both locally and as a scheduled job on Modal that runs once per day.

**{iris, titanic}-batch-inference-pipeline.py**

The batch-inference pipeline is for displaying the historical record of the model along with running a prediction on the batch data of the feature group. With each feature-pipeline-daily run a new synthetic flower/passenger is added to the feature group. The batch-inference pipeline runs inference on the entire batch of data which includes the new synthetic data and displays the predicted vs actual label of that synthetic data. The batch-inference pipeline also displays the historical record of predictions as well as the confusion matrix over the whole set of the data predicted. These get updated with each run that includes the new synthetic dataset as well.

The batch-inference pipeline can run both locally and as a scheduled job on Modal that runs once per day.

## Setup
For the initial part of the lab, the code for the Iris Flower prediction service is provided as part of the lab and we are required to connect the pipelines across the service providers and deploy the same to our own Hugging Face Space. For the second part of the lab, we build similar pipelines for the Titanic Survivor dataset where we predict the survival of a synthetically generated passenger. Both pipelines contain an interactive and a batch historical interface.

## Iris

_User interface links (Hugging Face)_
- Dashboard
    - https://huggingface.co/spaces/aykhazanchi/iris-monitoring
    - https://huggingface.co/spaces/rscolati/iris-monitoring
- Interactive
    - https://huggingface.co/spaces/aykhazanchi/iris
    - https://huggingface.co/spaces/rscolati/iris
- Code
    - Source code in `./iris`

### Dataset preparation and model
The implementation of the Iris flower prediction was used as provided.

## Titanic

_User interface links (Hugging Face)_
- Dashboard
    - https://huggingface.co/spaces/aykhazanchi/titanic-monitoring
    - https://huggingface.co/spaces/rscolati/titanic-monitoring
- Interactive
    - https://huggingface.co/spaces/aykhazanchi/titanic
    - https://huggingface.co/spaces/rscolati/titanic
- Code
    - Source code in `./titanic`

### Dataset preparation
After some initial testing we chose to use the following features
* Passenger age (binned using the following thresholds: 20, 25, 29, 30, 40)
* Passenger sex (converted to a numeric value, 0 for male or 1 for female)
* Passenger class (1st, 2nd, or 3rd)
* Passenger embarkation port (converted to a numeric value, 0 for “S”, 1 for “C”, or 2 for “Q”)

Missing values for the passengers’ ages and for the embarkation port were imputed using the mean and mode, respectively.

### Model
We use a RandomForestClassifier to train our data on the X_train set and predict the accuracy of the training against the y_train set. We considered the use of other models and settled on using RandomForestClassifier as it gave us a high enough accuracy in comparison.
