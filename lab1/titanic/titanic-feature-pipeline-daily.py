import modal

LOCAL = False


def generate_passenger(survived, passenger_id):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import numpy as np
    import random

    bins = [-np.infty, 20, 25, 30, 40, np.infty] # use same bins as in feature definition!
    age_bin_min = 0
    age_bin_max = len(bins) - 1

    random_age_bin = random.randint(age_bin_min, age_bin_max)
    random_sex = random.randint(0, 1)
    random_embarked = random.randint(0, 2)
    random_pclass = random.randint(1, 3)

    passenger_df = pd.DataFrame({"passengerid": [passenger_id],
                                 "age": [random_age_bin],
                                 "sex": [random_sex],
                                 "embarked": [random_embarked],
                                 "pclass": [random_pclass]})

    passenger_df['survived'] = survived
    return passenger_df


def get_random_titanic_passenger(passenger_id):
    """
    Returns a DataFrame containing one random passenger
    """
    import random

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.randint(0, 1)
    if pick_random == 1:
        passenger_df = generate_passenger(1, passenger_id)
        print("Survived")
    else:
        passenger_df = generate_passenger(0, passenger_id)
        print("Didn't survive")

    return passenger_df


def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)

    # get max id
    titanic_df = titanic_fg.read()
    passenger_id = titanic_df['passengerid'].max() + 1

    passenger_df = get_random_titanic_passenger(passenger_id)
    titanic_fg.insert(passenger_df, write_options={"wait_for_job": False})


if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        with stub.run():
            f()
            