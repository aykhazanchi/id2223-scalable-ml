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


def titanic(age, sex, pclass):
    input_list = []
    
    # Bin input age to bin index of range
    if age > 0 and age <= 20:
        input_list.append(0)
    elif age > 20 and age <= 50:
        input_list.append(1)
    elif age > 50 and age <= 75:
        input_list.append(2)
    elif age > 75:
        input_list.append(3)
    else:
        input_list.append(0) # negative age changes to zero

    input_list.append(int(sex)) # value returned by dropdown is index of option selected
    input_list.append(int(pclass+1)) # index starts at 0 so increment by 1

    print(input_list)
    # 'res' is a list of predictions returned as the label.
    #res = model.predict(np.asarray(input_list).reshape(1, -1), ntree_limit=model.best_ntree_limit)  # for xgboost
    print(np.asarray(input_list).reshape(1, -1))
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    print(res[0]) # 0/1 
    # below is just for testing
    if res[0] == 0: #ded
        passenger_url = "https://media.istockphoto.com/id/157612035/sv/foto/shipwreck.jpg?s=612x612&w=0&k=20&c=BSVml8_SqgvSmEijAprhniyp_Wa_l5qIIVIxhmmBgBQ="
    else:
        passenger_url = "https://i.chzbgr.com/full/5420028160/hD88BD9FE/like-a-boss"
    img = Image.open(requests.get(passenger_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Passenger Survival Predictive Analytics",
    description="Experiment with some passenger features to predict whether your passenger would have survived or not.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1, label="Age"),
        gr.inputs.Dropdown(choices=["Male", "Female"], type="index", label="Sex"),
        gr.inputs.Dropdown(choices=["Class 1","Class 2","Class 3"], type="index", label="Pclass"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

