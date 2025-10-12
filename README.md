# Speechcatcher

This is a Python utility to interface [Speechcatcher EspNet2 models](https://huggingface.co/speechcatcher). You can transcribe media files and use the utility for live transcription. All models are trained end-to-end with punctuation - the ASR model is able to output full text directly, without the need for punctuation reconstruction. Speechcatcher runs fast on CPUs and does not need a GPU to transcribe your audio.

The current focus is on German ASR. But more models will follow - stay tuned!

![Speechcatcher live recognition example](https://github.com/speechcatcher-asr/speechcatcher/raw/main/speechcatcher_de_live.gif)

## News

* 8/21/2025. New in version 0.4.2: Python3.13 compatibilty, made speechcatcher-server compatible with the new websockets>=14 API (see also https://websockets.readthedocs.io/en/stable/howto/upgrade.html).

* 1/7/2025. New in version 0.4.1: new and improved dynamic endpointing, improved error messages.

* 8/19/2024. New in version 0.4.0: Speechcatcher now has a websocket server (speechcatcher_server) for live transcription.

* 6/25/2024. New in version 0.3.2: Speechcatcher is now Python 3.12 compatible! Under certain conditions some input files would produce an error on the last segment, this now fixed in this version.

* 12/15/2023. New in version 0.3.1: Support for timestamps on the token level. Speechcatcher is now using its own [espnet_streaming_decoder](https://github.com/speechcatcher-asr/espnet_streaming_decoder) instead of using [espnet](https://github.com/espnet/espnet) directly, to make dependencies leaner and to enable token timestamps with streaming models. Speechcatcher does not require a full Espnet installation anymore. It also uses a forked version of [espnet_model_zoo](https://github.com/speechcatcher-asr/espnet_model_zoo), so that model downloads are only checked online if a local cache copy isn't available.

## Installation instructions:

Install portaudio and a few other dependencies, on Mac:

    brew install portaudio ffmpeg git git-lfs
    
on Linux (Ubuntu 24.04):
    
    sudo apt-get install portaudio19-dev python3.12-dev ffmpeg libhdf5-dev git git-lfs build-essential

on Linux (Fedora):
  
    sudo dnf install portaudio-devel python3 python3-pip python3-devel ffmpeg hdf5-devel git git-lfs gcc gcc-c++ make automake autoconf

For a system-wide and global installation, simply do:

    pip3 install git+https://github.com/speechcatcher-asr/speechcatcher

## Virtual environment

If you prefer an installation in a virtual environment, create one first. For example with python3.12:

    virtualenv -p python3.12 speechcatcher_env

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

or to launch a Vosk compatible websocket server for live transcription on ws://localhost:2700/ 

    speechcatcher_server --vosk-output-format --port 2700

All required model files are downloaded automatically and placed into a ".cache" directory.

## Use speechcatcher in your Python code

To use speechcatcher in your Python script, you need to import the speechcatcher package and use the recognize function in a '__main__' guarded block. Here is a complete example, that reads a 16kHz mono wav and outputs the recognized text:

    from speechcatcher import speechcatcher
    import numpy as np
    from scipy.io import wavfile
    
    # you need to run speechcatcher in a '__main__' guarded block:
    if __name__ == '__main__':
        short_tag = 'de_streaming_transformer_xl'
        speech2text = speechcatcher.load_model(speechcatcher.tags[short_tag])
    
        wav_file = 'input.wav'
        rate, audio_data = wavfile.read(wav_file)
        speech = audio_data.astype(np.int16)
    
        print(f"Sample Rate: {rate} Hz")
        print(f"Audio Shape: {audio_data.shape}")
    
        # speech is a numpy array of dtype='np.int16' (16bit audio with 16kHz sampling rate)
        complete_text, paragraphs = speechcatcher.recognize(speech2text, speech, rate, quiet=True, progress=False)
    
        # complete_text is a string with the complete decoded text
        print(complete_text)
        
        # -> Faust. Eine Tragödie von Johann Wolfgang von Goethe. Zueignung. Ihr naht euch wieder, schwankende Gestalten...

        # paragraphs contains a list of paragraphs with additional information, such as start and end position,
        # token and token_timestamps (upper bound, in seconds)
        print(paragraphs)
        
        # -> [{'start': 0, 'end': 44.51, 'text': 'Faust. Eine Tragödie von Johann Wolfgang von Goethe. Zueignung. Ihr naht euch wieder, schwankende Gestalten...', 'tokens': ['▁F', 'aus', 't', '.', '▁Ein', 'e', '▁Tra', 'g', 'ö', 'di', 'e', '▁von', '▁Jo', 'ha', 'n', 'n', '▁Wo', 'l', 'f', 'gang', '▁von', '▁G', 'o', 'et', 'he', '.', '▁Zu', 'e', 'ig', 'n', 'ung', '.', '▁I', 'hr', '▁', 'na', 'ht', '▁euch', '▁wieder', ',', '▁sch', 'wa', 'n', 'ken', 'de', '▁Ge', 'st', 'al', 'ten', '.', ... ],
        # 'token_timestamps': [1.666, 2.333, 2.333, 3.0, 3.0, 3.0, 3.0, 3.0, 3.666, 4.333, 4.333, 4.333, 5.0, 5.0, 5.0, 5.0, 5.0, 5.666, 5.666, 5.666, 6.333, 6.333, 6.333, 7.0, 7.666, 7.666, 7.666, 7.666, 8.333, 9.666, 9.666, 9.666, 9.666, 9.666, 9.666, 10.333, 10.333, 11.0, 11.666, 11.666, 11.666, 11.666, 11.666, 12.333, 12.333, 12.333, 13.0, 13.666, 14.333, 14.333, 14.333, 14.333, 14.333, 14.333, 14.333, ... ]}, ... ]


## Available models

| Acoustic model | Training data (hours) | Tuda-raw test WER (without LM) | CER |
| --- | --- | --- | --- |
| [de_streaming_transformer_m](https://huggingface.co/speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024) | 13k | 11.57 | 3.38 |
| [de_streaming_transformer_l](https://huggingface.co/speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_l_raw_de_bpe1024) | 13k | 9.65 | 2.76 |
| [de_streaming_transformer_xl](https://huggingface.co/speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024) | 26k | 8.5 | 2.44 | 
| --- | --- | --- | --- |
| [whisper large](https://huggingface.co/openai/whisper-large-v2) | ? | coming | soon! | 

Note: Tuda-de-raw results are based on raw lowercased tuda-de test utterances without the normalization step. It may not be directly comparable to regular tuda-de results.

## Speechcatcher CLI parameters

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
                            Choose a model: de_streaming_transformer_m, de_streaming_transformer_l or de_streaming_transformer_xl
      -d DEVICE, --device DEVICE
                            Computation device. Either 'cpu' or 'cuda'. Note: Mac M1 / mps support isn't available yet.
      --lang LANGUAGE       Explicitly set language, default is empty = deduct language from model tag
      -b BEAMSIZE, --beamsize BEAMSIZE
                            Beam size for the decoder
      --quiet               No partial transcription output when transcribing a media file
      --no-progress         Show no progress bar when transcribing a media file
      --no-exception-on-overflow
                            Do not abort live recognition when encountering an input overflow.
      --save-debug-wav      Save recording to debug.wav, only applicable to live decoding
      --num-threads NUM_THREADS
                            Set number of threads used for intraop parallelism on CPU in pytorch.
      -n NUM_PROCESSES, --num-processes NUM_PROCESSES
                            Set number of processes used for processing long audio files in parallel (the input file needs to be long enough). If set to -1, use multiprocessing.cpu_count() divided by two.

## Speechcatcher websocket parameters

    usage: speechcatcher_server [-h] [--host HOST] [--port PORT] [--model {de_streaming_transformer_m,de_streaming_transformer_l,de_streaming_transformer_xl}] [--device {cpu,cuda}] [--beamsize BEAMSIZE]
                                [--cache-dir CACHE_DIR] [--format {wav,mp3,mp4,s16le,webm,ogg,acc}] [--pool-size POOL_SIZE] [--vosk-output-format]

    Speechcatcher WebSocket Server for streaming ASR

    options:
      -h, --help            show this help message and exit
      --host HOST           Host for the WebSocket server
      --port PORT           Port for the WebSocket server
      --model {de_streaming_transformer_m,de_streaming_transformer_l,de_streaming_transformer_xl}
                            Model to use for ASR
      --device {cpu,cuda}   Device to run the ASR model on ('cpu' or 'cuda')
      --beamsize BEAMSIZE   Beam size for the decoder
      --cache-dir CACHE_DIR
                            Directory for model cache
      --format {wav,mp3,mp4,s16le,webm,ogg,acc}
                            Audio format for the input stream
      --pool-size POOL_SIZE
                            Number of speech2text instances to preload
      --vosk-output-format  Enable Vosk-like output format

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

Speechcatcher was gracefully funded by

<a href="https://media-tech-lab.com">Media Tech Lab by Media Lab Bayern</a> (<a href="https://github.com/media-tech-lab">@media-tech-lab</a>)

<a href="https://media-tech-lab.com">
    <img src="https://raw.githubusercontent.com/media-tech-lab/.github/main/assets/mtl-powered-by.png" width="240" title="Media Tech Lab powered by logo">
</a>


<a href="https://wordcab.com">Wordcab</a> (<a href="https://github.com/info-wordcab">@info-wordcab</a>)

<a href="https://wordcab.com">
    <img src="https://raw.githubusercontent.com/speechcatcher-asr/.github/refs/heads/main/wordcab_logo.webp" width="240" title="Wordcab logo">
</a>
