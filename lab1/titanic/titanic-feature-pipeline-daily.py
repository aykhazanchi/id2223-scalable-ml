import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def get_titanic_passenger(survived):
    """
    Returns a DataFrame containing one random Titanic passenger
    """
    import pandas as pd
    import random

    age_bin_min=0
    age_bin_max=3
    random_passengerid = random.randint(400,500)
    random_age_bin = random.int(age_bin_min, age_bin_max);

    random_sex = random.randint(0,1)
    random_pclass = random.randint(1,3)

    titanic_df = pd.DataFrame({ "passengerid": [random_passengerid],
                                "age": [random_age_bin],
                                "sex": [random_sex],
                                "pclass": [random_pclass],
                            })
    titanic_df['Survived'] = survived

    return titanic_df


def get_random_titanic_passenger():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.randint(0,1)
    if pick_random == 1:
        passenger_df = get_titanic_passenger(1)
        print("Survived")
    else:
        passenger_df = get_titanic_passenger(0)
        print("Didn't survive")
    
    return passenger_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_titanic_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=2)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
