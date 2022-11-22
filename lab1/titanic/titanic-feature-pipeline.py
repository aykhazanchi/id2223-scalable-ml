import hopsworks
import pandas as pd
import numpy as np

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

# Drop unnecessary columns
titanic_df = titanic_df.drop(columns=['Fare', 'Cabin', 'Name',
                                      'Parch', 'Ticket', 'SibSp'])

# Make gender numeric
titanic_df['Sex'] = titanic_df['Sex'].replace(['male', 'female'], [0, 1])

# Fill missing embarked values and make numeric
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
titanic_df['Embarked'] = titanic_df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])

# Interpolate missing Age values and create bins
bins = [-np.infty, 20, 25, 29, 30, 40, np.infty]
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Age'] = pd.cut(x=titanic_df['Age'], bins=bins, labels=False)

titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    primary_key=["PassengerId", "Age", "Sex", "Pclass", "Embarked"],
    version=1,
    description="Titanic dataset")

titanic_fg.insert(titanic_df, write_options={"wait_for_job": False})
