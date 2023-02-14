# Note: the implementation follows the espnet example note book released here: https://github.com/espnet/notebook/blob/master/espnet2_streaming_asr_demo.ipynb
# The notebook ( Apache-2.0 license ) was released with the clear intention of sharing how to use streaming models with EspNet2

import os
import sys
import argparse
import hashlib
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from espnet_model_zoo.downloader import ModelDownloader
import numpy as np
import wave
import pyaudio
import wavefile
import ffmpeg
import torch
from tqdm import tqdm

tags = {
"de_streaming_transformer_m" : "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024" }

device = 'cpu'
#device = 'mps'

# ensure that the directory for the path f exists
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# Load the espnet model with the given tag
def load_model(tag, beam_size=10, quiet=False):
    espnet_model_downloader = ModelDownloader(".cache/espnet")
    return Speech2TextStreaming(**espnet_model_downloader.download_and_unpack(tag, quiet=quiet),
        device=device, token_type=None, bpemodel=None,
        maxlenratio=0.0, minlenratio=0.0, beam_size=10, ctc_weight=0.3, lm_weight=0.0,
        penalty=0.0, nbest=1, disable_repetition_detection=True,
        decoder_text_length_limit=0, encoded_feat_length_limit=0
    )

# Convert input file to 16 kHz mono
def convert_inputfile(filename, outfile_wav):
    return (
        ffmpeg.input(filename)
        .output(outfile_wav, acodec='pcm_s16le', ac=1, ar='16k')
        .run(quiet=True, overwrite_output=True))

# Uses ANSI esc codes to delete previous lines. Resets the cursor to the beginning of an empty first line.
# see https://stackoverflow.com/questions/19596750/is-there-a-way-to-clear-your-printed-text-in-python
# and also "Everything you never wanted to know about ANSI escape codes"
# https://notes.burke.libbey.me/ansi-escape-codes/

def delete_multiple_lines(n=1):
    """Delete the last n lines in ."""
    for _ in range(n):
        sys.stdout.write("\x1b[2K")  # delete the last line
        sys.stdout.write("\x1b[1A")  # cursor up one line
    sys.stdout.write('\n\r')

def progress_output(text, prev_lines = 0):
    lines=['']
    last_i=''
    for i in text:
        if len(lines[-1]) > 100:
            if last_i==' ' or last_i=='.' or last_i=='?' or last_i=='!': 
                lines.append('')
        lines[-1] += i
        last_i = i 

    delete_multiple_lines(n=prev_lines)
    sys.stdout.write('\n'.join(lines))

    prev_lines = len(lines)
    sys.stdout.flush()

    return prev_lines

# using the model in 'speech2text', transcribe the path in 'media_path'
# quiet mode: don't output partial transcriptions
# progress mode: output transcription progress

def recognize(speech2text, media_path, quiet=False, progress=False):
    ensure_dir('.tmp/')
    wavfile_path = '.tmp/' + hashlib.sha1(args.inputfile.encode("utf-8")).hexdigest() + '.wav'
    convert_inputfile(media_path, wavfile_path)

    with wave.open(wavfile_path, 'rb') as wavfile_in:
        ch=wavfile_in.getnchannels()
        bits=wavfile_in.getsampwidth()
        rate=wavfile_in.getframerate()
        nframes=wavfile_in.getnframes()
        buf = wavfile_in.readframes(-1)
        data=np.frombuffer(buf, dtype='int16')
    
    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    sim_chunk_length = 8192 #640*4 #400
    speech_len = len(speech)

    prev_lines = 0

    if sim_chunk_length > 0:
        for i in tqdm(range(speech_len//sim_chunk_length), disable= not progress):
            results = speech2text(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                if not (quiet or progress):
                    prev_lines = progress_output(nbests[0], prev_lines)
            else:
                if not (quiet or progress):
                    prev_lines = progress_output("", prev_lines)

        results = speech2text(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
    else:
        results = speech2text(speech, is_final=True)
    nbests = [text for text, token, token_int, hyp in results]
    prev_lines = progress_output(nbests[0], prev_lines)

    print('\n')

    trans_file = media_path + '.txt'
    with open(trans_file, 'w') as trans_file_out:
        trans_file_out.write(nbests[0])

    print(f'Wrote transcription to {trans_file}.')
    os.remove(wavfile_path) 

# List all available microphones on this system
def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

# Stream audio data from a microphone to an espnet model
# Chunksize should be atleats 6400 for a lookahead of 16 frames 
def recognize_microphone(speech2text, tag, record_max_seconds=120, channels=1, recording_format=pyaudio.paInt16,
                         samplerate=16000, chunksize=8192, save_debug_wav=False):
    list_microphones()
    blocks=[]

    p = pyaudio.PyAudio()
    stream = p.open(format=recording_format, channels=channels, rate=samplerate, input=True, frames_per_buffer=chunksize)
    print(f'Model {tag} fully loaded, starting live transcription with your microphone.')

    n_best_lens = []
    finalize_update_iters = 10
    prev_lines = 0

    for i in range(0,int(samplerate/chunksize*record_max_seconds)+1):
        data=stream.read(chunksize)
        data=np.frombuffer(data, dtype='int16')
        if save_debug_wav:
            blocks.append(data)
        data=data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
        if i==int(samplerate/chunksize*record_max_seconds):
            results = speech2text(speech=data, is_final=True)
            break

        # Simple endpointing: Here we determine if no update happend in the last finalize_update_iters iterations
        # by checking the n (finalize_update_iters) lengths of the partial text output.
        # If all n previous lengths are the same, we finalize the ASR output and start a new utterance.

        if len(n_best_lens) < finalize_update_iters:
            finalize_iteration = False
        else:
            if all(x == n_best_lens[-1] for x in n_best_lens[-10:]):
                finalize_iteration = True
                n_best_lens = []
            else:
                finalize_iteration = False     

        results = speech2text(speech=data, is_final=finalize_iteration)

        if results is not None and len(results) > 0:
            nbests = [text for text, token, token_int, hyp in results]
            nbest_len = len(nbests[0])
            n_best_lens += [nbest_len]

            text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
            prev_lines = progress_output(nbests[0], prev_lines)
        else:
            prev_lines = progress_output("", prev_lines)

        if finalize_iteration:
            sys.stderr.write('\n')
            prev_lines = 0

    nbests = [text for text, token, token_int, hyp in results]
    prev_lines = progress_output(nbests[0], prev_lines)

    # Write debug wav as output file (will only be executed after shutdown)
    if save_debug_wav:
        print("Saving debug output...")
        wavefile.write("debug.wav", samplerate, np.concatenate(blocks, axis=None))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Speechcatcher utility to decode speech with speechcatcher espnet models.')
    parser.add_argument('-l', '--live-transcription', dest='live', help='Use microphone for live transcription', action='store_true')
    parser.add_argument('-m', '--model', dest='model', default='de_streaming_transformer_m', help='Choose the model file', type=str)
    parser.add_argument('-b','--beamsize', dest='beamsize', help='Beam size for the decoder', type=int, default=10)
    parser.add_argument('--quiet', dest='quiet', help='No partial transcription output when transcribing a media file', action='store_true')
    parser.add_argument('--progress', dest='progress', help='Show progress when transcribing a media file', action='store_true')
    parser.add_argument('inputfile', nargs='?', help='Input media file', default='')

    args = parser.parse_args()

    if args.model not in tags:
        print(f'Model {args.model} is not a valid model!')
        print('Options are:', ', '.join(tags.keys()))

    tag = tags[args.model]
    speech2text = load_model(tag=tag, beam_size=args.beamsize, quiet=args.quiet or args.progress)

    args = parser.parse_args()
    
    if args.live:
        recognize_microphone(speech2text, tag)
    elif args.inputfile != '':
        recognize(speech2text, args.inputfile, quiet=args.quiet, progress=args.progress)
    else:
        parser.print_help()
