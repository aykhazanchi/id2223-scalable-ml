import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

# Drop unnecessary columns
titanic_df = titanic_df.drop(columns=['Fare', 'Cabin', 'Embarked', 'Name',
                                      'Parch', 'Ticket', 'SibSp'])

# Make gender numeric (could be categorical?)
titanic_df['Sex'] = titanic_df['Sex'].replace(['male', 'female'], [0, 1])

# Interpolate missing Age values
# titanic_df['Age'] = titanic_df['Age'].interpolate()
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

# Bin the age data and create indexes
titanic_df['Age'] = pd.cut(x=titanic_df['Age'],
                           bins=[0, 20, 50, 75, 100], labels=False)

titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    primary_key=["PassengerId", "Age", "Sex", "Pclass"],
    version=1,
    description="Titanic dataset")

titanic_fg.insert(titanic_df, write_options={"wait_for_job": False})
