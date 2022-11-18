import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns


titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

# Drop columns
titanic_df.drop(columns=['Fare','Cabin','Embarked','Name','Parch','Ticket','SibSp'], inplace=True)
# Make gender numeric?
titanic_df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

# Interpolate missing Age values
titanic_df.interpolate(inplace=True)

# Bin the age data and create indexes
titanic_df['Age'] = pd.cut(x=titanic_df['Age'], bins=[0,20,50,75,100], labels=False)

print(titanic_df.head())

y = titanic_df.pop('Survived')
X = titanic_df

# You can read training data, randomly split into train/test sets of features (X) and labels (y)        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# In general, a small learning rate and large number of estimators = more accurate XGBoost models
# Reference for tuning: https://www.kaggle.com/code/alexisbcook/xgboost?scriptVersionId=79127842&cellId=9
# Train our model with the XGBoost algorithm using our features (X_train) and labels (y_train)
#model = XGBClassifier(n_estimators=500, learning_rate=0.4)
#model.fit(X_train, y_train.values.ravel(), early_stopping_rounds=50, eval_set=[(X_test, y_test)], verbose=False)
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train.values.ravel())


# Evaluate model performance using the features from the test set (X_test)
y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)

# Compare predictions (y_pred) with the labels in the test setx (y_test)
metrics = classification_report(y_test, y_pred, output_dict=True)
results = confusion_matrix(y_test, y_pred)

# Create the confusion matrix as a figure, we will later store it as a PNG image file
df_cm = pd.DataFrame(results, ['True Survived', 'True Died'],
                        ['Pred Survived', 'Pred Died'])
cm = sns.heatmap(df_cm, annot=True)
fig = cm.get_figure()

print(X_test.head())
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print(predictions)

print(accuracy)

d = {'Pclass':[3, 3, 1, 1], 'Sex':[0, 0, 0, 1], 'Age':[22.0, 52.0, 80.0, 10.0], 'SibSp':[1, 1, 1, 0]}
df = pd.DataFrame(data=d)
print(model.predict(df))
