# Speechcatcher

This is a Python utility to interface [Speechcatcher EspNet2 models](https://huggingface.co/speechcatcher). You can transcribe media files and use the utility for live transcription. All models are trained end-to-end with punctuation - the ASR model is able to output full text directly, without the need for punctuation reconstruction. Speechcatcher runs fast on CPUs and does not need a GPU to transcribe your audio.

The current focus is on German ASR. But more models will follow - stay tuned!

![Speechcatcher live recognition example](https://github.com/speechcatcher-asr/speechcatcher/raw/main/speechcatcher_de_live.gif)

## Installation instructions:

Install portaudio, on Mac:

    brew install portaudio ffmpeg
    
on Linux:
    
    sudo apt-get install portaudio19-dev python3.10-dev ffmpeg

For a system-wide and global installation, simply do:

    pip3 install git+https://github.com/speechcatcher-asr/speechcatcher

## Virtual environment

If you prefer an installation in a virtual environment, create one first:

    virtualenv -p python3.10 speechcatcher_env

Note, if you get "-bash: virtualenv: command not found", install virtualenv through pip:  

    #sudo pip3 install virtualenv 

Activate it:

    source speechcatcher_env/bin/activate

Then install speechcatcher:

    pip3 install git+https://github.com/speechcatcher-asr/speechcatcher

## Run speechcatcher from the command line

After you have succesfully installed speechcatcher, you can decode any media file with:

    speechcatcher media_file.mp4

or to transcribe data live from your microphone:

    speechcatcher -l

All required model files are downloaded automatically and placed into a ".cache" directory.

## Use speechcatcher in your Python code

To use speechcatcher in your Python script import the speechcatcher package and use the recognize function:

    from speechcatcher import speechcatcher
    short_tag = 'de_streaming_transformer_xl'
    speech2text = speechcatcher.load_model(speechcatcher.tags[short_tag])
    
    # speech is a numpy array of dtype='np.int16' (audio with 16kHz sampling rate)
    complete_text, paragraphs, paragraphs_tokens, paragraph_hyps, segments_start_end_in_seconds = speechcatcher.recognize(speech2text, speech, rate, quiet=True, progress=False)

    # or use the recognize_file function, where media_path can be in any fileformat that ffmpeg supports
    # result object is: {"complete_text":complete_text, "segments_start_end_in_seconds":segments_start_end_in_seconds,
    #                    "segments":paragraphs, "segment_tokens":paragraphs_tokens}
    result = recognize_file(speech2text, media_path, output_file='', quiet=True, progress=False, num_processes=4)

## Available models

| Acoustic model | Training data (hours) | Tuda-raw test WER (without LM) | CER |
| --- | --- | --- | --- |
| de_streaming_transformer_m | 13k | 11.57 | 3.38 |
| de_streaming_transformer_l | 13k | 9.65 | 2.76 |
| de_streaming_transformer_xl | 26k | 8.5 | 2.44 | 
| --- | --- | --- | --- |
| whisper large | ? | coming | soon! | 

Note: Tuda-de-raw results are based on raw tuda-de test utterances without the normalization step. It may not be directly comparable to regular tuda-de results.

## Speechcatcher CLI parameters

    usage: speechcatcher [-h] [-l] [-t MAX_RECORD_TIME] [-m MODEL] [-d DEVICE] [--lang LANGUAGE] [-b BEAMSIZE] [--quiet] [--no-progress] [--save-debug-wav] [--num-threads NUM_THREADS] [-n NUM_PROCESSES] [inputfile]

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

See [speechcatcher-data](https://github.com/speechcatcher-asr/speechcatcher-data) for code and more info on replicating the training process.

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
