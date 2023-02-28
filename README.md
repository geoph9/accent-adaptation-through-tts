# Accent Adaptation Using Synthesized Data

This is the public repository containing the scripts used to process and generate synthesized
data, and use them to fine-tune the `wav2vec2.0-base` model. More info to be published soon.

## Data

We used the common voice (version 8, English-only) and L2-Arctic databases. 

## Synthesised Data (TTS)

The code under `data_synth` was used to generate the synthesized accented audio data.
The main functionality of these scripts is taken from the
[`Multilingual_TTS_Augmentation`](https://github.com/n8rob/Multilingual_TTS_Augmentation) repo. The 
current repo will likely not follow any changes from the original repo, so if you want to use an 
up-to-date version of this work you should separately clone `n8rob`'s repo.