# Speechcatcher

This is a Python utility to interface [Speechcatcher EspNet2 models](https://huggingface.co/speechcatcher). You can transcribe media files and use the utility for live transcription. All models are trained end-to-end with punctuation - the ASR model is able to output full text directly, without the need for punctuation reconstruction.

Our first model is for German, trained on 13k hours of speech. More models will follow - stay tuned!

![Speechcatcher live recognition example](https://github.com/speechcatcher-asr/speechcatcher/raw/main/speechcatcher_de_live.gif)

## Installation instructions:

Install portaudio, on Mac:

    brew install portaudio libsndfile
    
on Linux:
    
    sudo apt-get install portaudio19-dev

Create a virtual environment:

    virtualenv -p python3.10 speechcatcher_env

Note, if you get "-bash: virtualenv: command not found", install virtualenv through pip:  

    #sudo pip3 install virtualenv 

Activate it:

    source speechcatcher_env/bin/activate

Then install the requirements:

    pip3 install -r requirements.txt
   
Done! You can then run speechcatcher with:

    python3 speechcatcher.py media_file.mp4

or to transcribe data live from your microphone:

    python3 speechcatcher.py -l

All required model files are downloaded automatically and placed into a ".cache" directory.

## Speechcatcher training

Speechcatcher models are trained by using Whisper large as a teacher model:

![Speechcatcher Teacher/student training](https://github.com/speechcatcher-asr/speechcatcher/raw/main/speechcatcher_training.svg)

## Citation

If you use speechcatcher models in your research, for now just cite this repository:

    @misc{milde2023speechcatcher,
      author = {Milde, Benjamin},
      title = {Speechcatcher},
      year = {2023},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/speechcatcher-asr/speechcatcher}},
    }

## Sponsors

Speechcatcher is gracefully funded by

<a href="https://media-tech-lab.com">Media Tech Lab by Media Lab Bayern</a> (<a href="https://github.com/media-tech-lab">@media-tech-lab</a>)

<a href="https://media-tech-lab.com">
    <img src="https://raw.githubusercontent.com/media-tech-lab/.github/main/assets/mtl-powered-by.png" width="240" title="Media Tech Lab powered by logo">
</a>
