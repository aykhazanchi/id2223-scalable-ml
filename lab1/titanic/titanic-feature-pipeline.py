import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

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

titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    primary_key=["PassengerId","Age","Sex","Pclass"],
    version=2,
    description="Titanic dataset")

#titanic_fg.delete()

titanic_fg.insert(titanic_df, write_options={"wait_for_job": False}, overwrite=True)


#expectation_suite = ge.core.ExpectationSuite(expectation_suite_name="titanic_dimensions")    
#value_between(expectation_suite, "sepal_length", 4.5, 8.0)
#value_between(expectation_suite, "sepal_width", 2.1, 4.5)
#value_between(expectation_suite, "petal_length", 1.2, 7)
#value_between(expectation_suite, "petal_width", 0.2, 2.5)
#titanic_fg.save_expectation_suite(expectation_suite=expectation_suite, validation_ingestion_policy="STRICT")    
    

