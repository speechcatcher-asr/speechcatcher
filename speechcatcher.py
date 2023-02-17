# Note: the decoding implementation is inspired by the espnet example note book released here: https://github.com/espnet/notebook/blob/master/espnet2_streaming_asr_demo.ipynb
# However, the implementation here is substantically different and rewritten now
# Among other things, there is endpointing for the live and batch decoder. Threaded I/O for the microphone.
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
from scipy.io import wavfile
import ffmpeg
import torch
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from simple_endpointing import segment_wav 

tags = {
"de_streaming_transformer_m" : "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024" }

# ensure that the directory for the path f exists
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# Load the espnet model with the given tag
def load_model(tag, device='cpu' ,beam_size=10, quiet=False):
    espnet_model_downloader = ModelDownloader(".cache/espnet")
    return Speech2TextStreaming(**espnet_model_downloader.download_and_unpack(tag, quiet=quiet),
        device=device, token_type=None, bpemodel=None,
        maxlenratio=0.0, minlenratio=0.0, beam_size=beam_size, ctc_weight=0.3, lm_weight=0.0,
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
    
    assert(rate==16000)

    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    sim_chunk_length = 8192 #640*4 #400
    speech_len = len(speech)

    prev_lines = 0

    segments = segment_wav(wavfile_path)
    print(segments)
    utterance_text = ''
    complete_text = ''

    if sim_chunk_length > 0:
        for i in tqdm(range(speech_len//sim_chunk_length), disable= not progress):
            # first calculate in seconds, then mutiply with 100 to get the framepos (100 frames in one second.)
            frame_pos = ((i+1)*sim_chunk_length / rate ) * 100.
            if len(segments) > 0 and frame_pos >= segments[0][1]:
                is_final = True
                segments = segments[1:]
            else:
                is_final = False
            results = speech2text(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=is_final)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                utterance_text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                if not (quiet or progress):
                    prev_lines = progress_output(nbests[0], prev_lines)
            else:
                if not (quiet or progress):
                    prev_lines = progress_output("", prev_lines)
            if not (quiet or progress):
                if is_final:
                    sys.stdout.write('\n')
                    prev_lines = 0
                    # with endpointing, its likely that there is a pause between the segments
                    # here we actually check if the model thinks that this ending is also a sentence ending
                    # only add a parapgrah to the text output if model and end pointer agree on the segment boundary
                    utterance_is_completed = utterance_text.endswith('.') or utterance_text.endswith('?') or utterance_text.endswith('!')
                    complete_text += utterance_text + ('\n\n' if utterance_is_completed else ' ')
                    utterance_text = ''

        results = speech2text(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
    else:
        results = speech2text(speech, is_final=True)

    nbests = [text for text, token, token_int, hyp in results]
    prev_lines = progress_output(nbests[0], prev_lines)
    complete_text += nbests[0]

    print('\n')
    trans_file = media_path + '.txt'
    with open(trans_file, 'w') as trans_file_out:
        trans_file_out.write(complete_text)

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
                         samplerate=16000, chunksize=8192, save_debug_wav=False, exception_on_pyaudio_overflow=True,
                         finalize_update_iters=7):
    list_microphones()
    blocks=[]

    p = pyaudio.PyAudio()
    stream = p.open(format=recording_format, channels=channels, rate=samplerate, input=True, frames_per_buffer=chunksize)
    print(f'Model {tag} fully loaded, starting live transcription with your microphone.')

    n_best_lens = []
    prev_lines = 0
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        data_future = executor.submit(stream.read, chunksize, exception_on_overflow=exception_on_pyaudio_overflow)
        for i in range(0,int(samplerate/chunksize*record_max_seconds)+1):
            executor.submit

            data = data_future.result(timeout=2)
            data_future = executor.submit(stream.read, chunksize, exception_on_overflow=exception_on_pyaudio_overflow)

            data=np.frombuffer(data, dtype='int16')
            if save_debug_wav:
                blocks.append(data)
            
            #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
            data=data.astype(np.float16)/32767.0
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
                sys.stdout.write('\n')
                prev_lines = 0

        nbests = [text for text, token, token_int, hyp in results]
        prev_lines = progress_output(nbests[0], prev_lines)

    # Write debug wav as output file (will only be executed after shutdown)
    if save_debug_wav:
        print("\nSaving debug output...")
        wavfile.write("debug.wav", samplerate, np.concatenate(blocks, axis=None))

    print("\nMaximum recording time reached, exiting.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Speechcatcher utility to decode speech with speechcatcher espnet models.')
    parser.add_argument('-l', '--live-transcription', dest='live', help='Use microphone for live transcription', action='store_true')
    parser.add_argument('-t', '--max-record-time', dest='max_record_time', help='Maximum record time in seconds (live transcription).', type=float, default=120)
    parser.add_argument('-m', '--model', dest='model', default='de_streaming_transformer_m', help='Choose the model file', type=str)
    parser.add_argument('-d', '--device', dest='device', default='cpu', help="Computation device. Either 'cpu' or 'cuda'. Note: mac m1 / mps support isn't available yet.")    
    parser.add_argument('--lang', dest='language', default='', help='Explicity set language, default is empty = deduct languagefrom model tag', type=str)
    parser.add_argument('-b','--beamsize', dest='beamsize', help='Beam size for the decoder', type=int, default=10)
    parser.add_argument('--quiet', dest='quiet', help='No partial transcription output when transcribing a media file', action='store_true')
    parser.add_argument('--progress', dest='progress', help='Show progress when transcribing a media file', action='store_true')
    parser.add_argument('--save-debug-wav', dest='save_debug_wav', help='Save recording to debug.wav, only applicable to live decoding', action='store_true')

    parser.add_argument('inputfile', nargs='?', help='Input media file', default='')

    args = parser.parse_args()

    if args.model not in tags:
        print(f'Model {args.model} is not a valid model!')
        print('Options are:', ', '.join(tags.keys()))

    tag = tags[args.model]
    speech2text = load_model(tag=tag, device=args.device, beam_size=args.beamsize, quiet=args.quiet or args.progress)

    args = parser.parse_args()
    
    if args.live:
        recognize_microphone(speech2text, tag, record_max_seconds=args.max_record_time, save_debug_wav=args.save_debug_wav)
    elif args.inputfile != '':
        recognize(speech2text, args.inputfile, quiet=args.quiet, progress=args.progress)
    else:
        parser.print_help()
