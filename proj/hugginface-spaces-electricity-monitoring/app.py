import gradio as gr
import hopsworks

project = hopsworks.login()

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/images/df_ny_elec_recent.png", overwrite=True)
dataset_api.download("Resources/images/df_ny_elec_prediction.png", overwrite=True)
dataset_api.download("Resources/images/df_ny_elec_mae.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's prediction")
            input_img = gr.Image("df_ny_elec_prediction.png", elem_id="latest-prediction")
        with gr.Column():
            gr.Label("Recent Prediction History")
            input_img = gr.Image("df_ny_elec_recent.png", elem_id="recent-predictions")
            input_img = gr.Image("df_ny_elec_mae.png", elem_id="recent-predictions")


demo.launch()
