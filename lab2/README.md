# ID2223 - Lab 2

TODO: short description

## Interface

We implemented a simple interface which allows to create a transcript from an uploaded or recorded audio file using the deployed model.

The interface app is implemented in [huggingface-spaces-whisper-sv](./huggingface-spaces-whisper-sv), and available online.

Link(s)
- https://huggingface.co/spaces/rscolati/whisper-sv

## Notebooks (Implementation)

The data preparation and training are implemented in the following notebooks.
* [data_prep.ipynb](./data_prep.ipynb) -- Data preparation
* [whisper_sv.ipynb](./whisper_sv.ipynb) -- Training

We further used (some iteration of) the following notebook to push models and checkpoints to the Huggingface remote, since the builtin functionality had some issues.
* [push_to_hub.ipynb](./push_to_hub.ipynb) -- Manual push to HF hub (we had some issues with Git LFS, see notes below)

### Data Preparation

We followed the provided sample notebook mostly, the only changes we made was splitting the data preparation part from the training part to be able to re-use the prepared data sets.

We followed the main steps (data download, conversion, and splits) and created a compressed archive (gzipped tar file, `!tar -czvf "drive/MyDrive/common_voice.tar.gz" "common_voice"`) of the sets stored to Google drive.

### Training

TODO: short description of what the notebook does

Model training is WIP, the current iteration is available for testing at https://huggingface.co/rscolati/whisper-small-sv (should be public).

### Notes, problems, and alternative approaches

* We decided to split data preparation from training, since we had to resume from checkpoints often due to Colab limitations. This way we can prepare the data once and re-use it.
* Since push-to-hub seems broken (might be a Git LFS issue) we used Google Drive to store the output (checkpoints and models) as to not lose the state whenever Colab disconnects.
* The push-to-hub is done manually since Git seems to have problems when pushing from Google Drive.
* We lost a considerable amount of time due to not having enough resources from Colab on a free account, as alternatives we tried
    * Kaggle (would have required to run with a custom container since we had a lot of dependency version issues, also random disconnects)
    * Modal (kept timing out even when setting function timeout to 3h, which is the maximum)
* We tried to work with a "smaller" model (whisper-base, trained/tuned model with 2000 steps [here](https://huggingface.co/rscolati/whisper-base-sv)) but the time did not change considerably and the performance seemed worse (WER around 10% worse after comparable training)
