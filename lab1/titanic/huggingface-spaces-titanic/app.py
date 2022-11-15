import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(age, sex, pclass, sibsp):
    input_list = []
    input_list.append(age)
    input_list.append(sex)
    input_list.append(pclass)
    input_list.append(sibsp)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    passenger_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Passenger Survival Predictive Analytics",
    description="Experiment with passenger features to predict whether they would have survived or not.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Age"),
        gr.inputs.Number(default=1.0, label="Sex (0 = Male, 1 = Female)"),
        gr.inputs.Number(default=1.0, label="pclass"),
        gr.inputs.Number(default=1.0, label="SibSp"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

