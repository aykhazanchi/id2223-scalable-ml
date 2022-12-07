# ID2223 - Lab 2

TODO: short description

## Notebooks

The data preparation and training are implemented in the following notebooks
* [data_prep.ipynb](./data_prep.ipynb) -- Data preparation
* [whisper_sv.ipynb](./whisper_sv.ipynb) -- Training
* [push_to_hub.ipynb](./push_to_hub.ipynb) -- Manual push to HF hub (Git LFS has some issues, see notes below)

### Data Preparation

TODO: short description of what the notebook does

### Training

TODO: short description of what the notebook does

Model training is WIP, the current iteration is available for testing at https://huggingface.co/rscolati/whisper-small-sv (should be public).

### Notes and problems

* Since push-to-hub seems broken (might be a Git LFS issue), we split the pipeline in two parts, data preparation and training, and used Google Drive to store the output (checkpoints and models) as to not lose the state whenever Colab disconnects
* The push-to-hub is done manually since Git seems to have problems when pushing from Google Drive
* We lost a considerable amount of time due to not having enough resources from Colab on a free account, alternatives tried
    * Kaggle (would have required to run with a custom container since we had a lot of dependency version issues, also random disconnects)
    * Modal (kept timing out even when setting function timeout to 3h, which is the maximum)
* We tried to work with a "smaller" model (whisper-base, trained/tuned model with 2000 steps [here](https://huggingface.co/rscolati/whisper-base-sv)) but the time did not change considerably and the performance seemed worse (WER around 10% worse after comparable training)

## Interface

TODO: short description

Link(s)
- https://huggingface.co/spaces/rscolati/whisper-sv

Implemented in [huggingface-spaces-whisper-sv](./huggingface-spaces-whisper-sv)
