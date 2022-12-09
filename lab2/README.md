# ID2223 - Lab 2

The goal of the assignment was to fine-tune a pre-trained transformer model, Whisper, to transcribe Swedish language audio (or audio of our mother tongue) to text. We chose to transcribe the Swedish language.

## Interface

We implemented a simple interface which allows to create a transcript from an uploaded or recorded audio file using the (latest) deployed model.

The interface app is implemented in [huggingface-spaces-whisper-sv](./huggingface-spaces-whisper-sv), and available online.

### Link to the interactive UI
- https://huggingface.co/spaces/rscolati/whisper-sv

## Implementation (Notebooks)

The data preparation and training are implemented in the following notebooks.
* [data_prep.ipynb](./data_prep.ipynb) -- Data preparation
* [whisper_sv.ipynb](./whisper_sv.ipynb) -- Training

We further used an iteration of the following notebook to push models and checkpoints to the Huggingface remote, since the builtin functionality had some issues.
* [push_to_hub.ipynb](./push_to_hub.ipynb) -- Manual push to HF hub (we had some issues with Git LFS, see notes below)

### Data Preparation

We followed the provided sample notebook mostly, the only changes we made was splitting the data preparation part from the training part to be able to re-use the prepared data sets.

We followed the main steps (data download, conversion, and splits) and created a compressed archive (gzipped tar file, `!tar -czvf "drive/MyDrive/common_voice.tar.gz" "common_voice"`) of the sets stored to Google drive.

### Training

For the training we followed moslty the proposed implementation using Colab. We ran into several issues, mostly due to limited compute time and resources, and considered and tried some alternative approaches wrt. model size and alternative platforms (see [Notes, problems, and alternative approaches](#notes-problems-and-alternative-approaches)).

In the final implementation we changed the training parameters slightly, setting 2000 optimization steps and generating a checkpoint after every 500 steps. Model (and metrics during training) is available online, word error rate as measured during evaluation was 19.64%.

Link:
- https://huggingface.co/rscolati/whisper-small-sv/tensorboard

### Notes, problems, and alternative approaches

The main problem we encountered was the size of the dataset and model (and thus training time), and the time and resource limitations on free Google Colab accounts. Since we had to use checkpoints and split the training in separate runs, we made some changes to the original proposed notebook.
* We decided to split data preparation from training so we could prepare the data once and re-use it, storing it to Google Drive. Additionally, data preparation can be carried out using a CPU which allows us to divert free GPU allocations towards training.
* Since the builtin push-to-hub functionality seemed broken (might be a Git LFS issue) we used Google Drive to store the output (checkpoints and models) so as to not lose the state whenever Colab disconnected. Data was then pushed manually.

Since we lost a considerable amount of time due to Colab's resource quotas we explored some alternatives, namely Kaggle and Modal, but we decided to keep using Colab since we were running into similar and other issues with both platforms.
* Kaggle would have required us to run with a custom container since we had a lot of dependency version issues, and we experienced lots of random disconnects during training.
* Modal kept timing out even when setting function timeout to 3h, which is the maximum. 

We also tried to work with a smaller pre-trained model (whisper-base), but the required time did not change considerably and the performance seemed worse. The word error rate was around 10% worse after comparable training, a trained model (with 2000 steps) is available [here](https://huggingface.co/rscolati/whisper-base-sv).
