import gradio as gr
import hopsworks
from sklearn.metrics import mean_absolute_error

project = hopsworks.login()
fs = project.get_feature_store()

monitor_fg = fs.get_feature_group(name="ny_elec_predictions", version=1)
history_df = monitor_fg.read()
latest_prediction = history_df.iloc[-1]

y_pred = history_df['prediction']
y_test = history_df['actual']
mean_error = mean_absolute_error(y_test, y_pred)

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/images/df_ny_elec_recent.png", overwrite=True)

with gr.Blocks() as demo:
    gr.Label("Today's prediction")
    with gr.Row():
        with gr.Column():
            gr.Textbox(value="{}".format(latest_prediction['prediction_date']),
                       label="Prediction date")
        with gr.Column():
            gr.Textbox(value="{:.0f}MWh".format(latest_prediction['prediction']),
                       label="Predicted NY electricity demand")
    with gr.Row():
        with gr.Column():
            gr.Textbox(value="{}MWh".format(latest_prediction['actual']),
                       label="Actual demand")
        with gr.Column():
            gr.Textbox(value="{}MWh".format(latest_prediction['forecast_eia']),
                       label="EIA forecast")
    gr.Label("Recent Prediction History")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image("df_ny_elec_recent.png", elem_id="recent-predictions")
        with gr.Column():
            gr.Textbox(label="MAE for historical predictions",
                       value="{:.0f}MWh".format(mean_error))


demo.launch()
