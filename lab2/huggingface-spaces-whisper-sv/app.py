from transformers import pipeline
import gradio as gr

pipe = pipeline(model="rscolati/whisper-small-sv")


def transcribe(rec=None, file=None):
    if rec is not None:
        audio = rec
    elif file is not None:
        audio = file
    else:
        return "Provide a recording or a file."

    text = pipe(audio)["text"]
    return text


iface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", optional=True),
        gr.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper model.",
)


iface.launch()
