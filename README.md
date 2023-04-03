# Speechcatcher

This is a Python utility to interface [Speechcatcher EspNet2 models](https://huggingface.co/speechcatcher). You can transcribe media files and use the utility for live transcription. All models are trained end-to-end with punctuation - the ASR model is able to output full text directly, without the need for punctuation reconstruction.

The current focus is on German ASR. But more models will follow - stay tuned!

![Speechcatcher live recognition example](https://github.com/speechcatcher-asr/speechcatcher/raw/main/speechcatcher_de_live.gif)

## Installation instructions:

Install portaudio, on Mac:

    brew install portaudio ffmpeg
    
on Linux:
    
    sudo apt-get install portaudio19-dev python3.10-dev ffmpeg

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

To use speechcatcher in your Python script:

    import speechcatcher
    short_tag = 'de_streaming_transformer_m'
    speech2text = speechcatcher.load_model(speechcatcher.tags[short_tag])
    
    text = speechcatcher.recognize(speech2text, speech, rate, quiet=True, progress=False)

Currently, you would need to put your script into the same folder as speechcatcher.py, but this might be fixed in an upcoming release when speechcatcher is a proper Python module.

## Available models

| Acoustic model | Training data (hours) | Tuda test WER (without LM) | CER |
| --- | --- | --- | --- |
| de_streaming_transformer_m | 13k | 11.57 | 3.38 |
| de_streaming_transformer_l | 13k | 9.65 | 2.76 |
| de_streaming_transformer_xl | 26k | coming | soon! | 
| --- | --- | --- | --- |
| whisper large | ? | coming | soon! | 

## Speechcatcher CLI parameters

    usage: speechcatcher.py [-h] [-l] [-t MAX_RECORD_TIME] [-m MODEL] [-d DEVICE] [--lang LANGUAGE] [-b BEAMSIZE] [--quiet] [--no-progress] [--save-debug-wav] [--num-threads NUM_THREADS] [-n NUM_PROCESSES] [inputfile]

    Speechcatcher utility to decode speech with speechcatcher espnet models.

    positional arguments:
      inputfile             Input media file

    options:
      -h, --help            show this help message and exit
      -l, --live-transcription
                            Use microphone for live transcription
      -t MAX_RECORD_TIME, --max-record-time MAX_RECORD_TIME
                            Maximum record time in seconds (live transcription).
      -m MODEL, --model MODEL
                            Choose a model: de_streaming_transformer_m or de_streaming_transformer_l
      -d DEVICE, --device DEVICE
                            Computation device. Either 'cpu' or 'cuda'. Note: Mac M1 / mps support isn't available yet.
      --lang LANGUAGE       Explicitly set language, default is empty = deduct language from model tag
      -b BEAMSIZE, --beamsize BEAMSIZE
                            Beam size for the decoder
      --quiet               No partial transcription output when transcribing a media file
      --no-progress         Show no progress bar when transcribing a media file
      --save-debug-wav      Save recording to debug.wav, only applicable to live decoding
      --num-threads NUM_THREADS
                            Set number of threads used for intraop parallelism on CPU in pytorch.
      -n NUM_PROCESSES, --num-processes NUM_PROCESSES
                            Set number of processes used for processing long audio files in parallel (the input file needs to be long enough).
                            If set to -1, use multiprocessing.cpu_count() divided by two.

## Speechcatcher training

Speechcatcher models are trained by using Whisper large as a teacher model:

![Speechcatcher Teacher/student training](https://github.com/speechcatcher-asr/speechcatcher/raw/main/speechcatcher_training.svg)

See ![https://github.com/speechcatcher-asr/speechcatcher-data](speechcatcher-data) for code and more info on replicating the training process.

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
