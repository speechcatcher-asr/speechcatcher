# Speechcatcher

This is a Python utility to interface [Speechcatcher EspNet2 models](https://huggingface.co/speechcatcher). You can transcribe media files and use the utility for live transcription. All models are trained end-to-end with punctuation - the ASR model is able to output full text directly, without the need for punctuation reconstruction.

Our first model is for German, trained on 13k hours of speech. More models will follow - stay tuned!

## Installation instructions:

Create a virtual environment:

    virtualenv -p python3.10 speechcatcher_env

Activate it:

    source speechcatcher_env/bin/activate

Then install the requirements:

    pip3 install -r requirements.txt
   
Done! You can then run speechcatcher with:

    python3 speechcatcher.py media_file.mp4

or to transcribe data live from your microphone:

    python3 speechcatcher.py -l

All required model files are downloaded automatically and placed into a ".cache" directory.
