# Note: the decoding implementation is inspired by the espnet example note book released here: https://github.com/espnet/notebook/blob/master/espnet2_streaming_asr_demo.ipynb
# However, the implementation here is substantically different and basically rewritten
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
import math

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from simple_endpointing import segment_wav 

tags = {
"de_streaming_transformer_m" : "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024",
"de_streaming_transformer_l" : "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_l_raw_de_bpe1024" }

# ensure that the directory for the path f exists
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# Load the espnet model with the given tag
def load_model(tag, device='cpu' ,beam_size=5, quiet=False):
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

# Output current hypothesis on the fly. Note that .
def progress_output(text, prev_lines = 0):
    lines=['']
    last_i=''
    for i in text:
        if len(lines[-1]) > 100:
            # make sure that we don'T
            if last_i==' ' or last_i=='.' or last_i=='?' or last_i=='!': 
                lines.append('')
        lines[-1] += i
        last_i = i 

    delete_multiple_lines(n=prev_lines)
    sys.stdout.write('\n'.join(lines))

    prev_lines = len(lines)
    sys.stdout.flush()

    return prev_lines

# This function makes sure the first letter of an utterance / paragraph is capitalized
def upperCaseFirstLetter(utterance_text):
    if len(utterance_text) > 0 and utterance_text[0].islower():
        utterance_text = utterance_text[0].upper() + utterance_text[1:]
    return utterance_text

# Checks if an utterance is completed
def is_completed(utterance):
    return utterance.endswith('.') or utterance.endswith('?') or utterance.endswith('!')

# Using the model in 'speech2text', transcribe the path in 'media_path'
# quiet mode: don't output partial transcriptions
# progress mode: output transcription progress

def recognize(speech2text, media_path, output_file='', quiet=False, progress=False):
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
    chunk_length = 8192
    speech_len = len(speech)

    prev_lines = 0

    segments = segment_wav(wavfile_path)
    segment_frame_pos_end = [segment[1] for segment in segments]

    max_i = (speech_len // chunk_length) + 1

    # the segments from segment_wav are in frame positions (100 frames per second)
    # with the given chunksize, we calculate the iterations here where we need to finalize
    # note: we do not finalize at the beginning, but at -1 to easily calculate start and end positions for the loop below
    segments_i = [-1]+[math.ceil((((f/100.)*rate) - chunk_length) / chunk_length) for f in segment_frame_pos_end]+[max_i]

    #print('finalize iterations:', segments_i)

    utterance_text = ''
    complete_text = ''
    paragraphs = []
    speech_len = len(speech)

    seg_num = 1
    seg_num_total = len(segments_i)
    for start, end in zip(segments_i[:-1], segments_i[1:]):
        for i in tqdm(range(start+1, end+1), disable=not progress, desc=f'Segment {seg_num}/{seg_num_total}'):
            speech_chunk_start = i * chunk_length
            speech_chunk_end = (i + 1) * chunk_length
            if speech_chunk_end > speech_len:
                speech_chunk_end = speech_len

            speech_chunk = speech[speech_chunk_start: speech_chunk_end]

            paragraphs, prev_lines = batch_recognize_inner_loop(speech2text, speech_chunk, i, paragraphs, prev_lines,
                                                                progress, quiet, rate, chunk_length,
                                                                utterance_text, is_final=i in segments_i)
        seg_num += 1

    complete_text = '\n\n'.join(paragraphs)

    print('\n')

    # Automatically generate output .txt name from media_path if it isnt set
    # media_path can also be an URL, in that case it needs special handling
    if output_file == '':
        if media_path.startswith('http://') or media_path.startswith('https://'):
            output_file = media_path.split('/')[-1] + '.txt'
        else:
            output_file = media_path + '.txt'

    with open(output_file, 'w') as output_file_out:
        output_file_out.write(complete_text)

    print(f'Wrote transcription to {output_file}.')
    os.remove(wavfile_path)


def batch_recognize_inner_loop(speech2text, speech_chunk, i, paragraphs, prev_lines, progress, quiet, rate,
                               chunk_length, utterance_text, is_final, debug_pos=False):

    # first calculate in seconds, then multiply with 100 to get the framepos (100 frames in one second.)
    frame_pos = ((i + 1) * chunk_length / rate) * 100.
    if is_final and debug_pos:
        print('is final @', f'{i}', f'{frame_pos}')

    results = speech2text(speech=speech_chunk, is_final=is_final)
    if results is not None and len(results) > 0:
        nbests = [text for text, token, token_int, hyp in results]
        utterance_text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
        if not (quiet or progress):
            prev_lines = progress_output(nbests[0], prev_lines)
    else:
        if not (quiet or progress):
            prev_lines = progress_output("", prev_lines)
        utterance_text = ''
        
    if is_final:
        prev_lines = 0

        # with endpointing, it's likely that there is a pause between the segments
        # here we actually check if the model thinks that this ending is also a sentence ending
        # only add a paragraph to the text output if model and end pointer agree on the segment boundary
        prev_utterance_is_completed = True
        if len(paragraphs) > 0:
            prev_utterance_is_completed = is_completed(paragraphs[-1])
        if prev_utterance_is_completed:
            # Make sure the paragraph starts with a capitalized letter
            utterance_text = upperCaseFirstLetter(utterance_text)
            paragraphs += [utterance_text]
        else:
            # might be in the middle of a sentence - append to the last (open) paragraph
            paragraphs[-1] += ' ' + utterance_text

        if not (quiet or progress) and is_completed(utterance_text):
            sys.stdout.write('\n')

    return paragraphs, prev_lines


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

            # Simple endpointing: Here we determine if no update happened in the last finalize_update_iters iterations
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
    parser.add_argument('-m', '--model', dest='model', default='de_streaming_transformer_l', help='Choose a model: de_streaming_transformer_m or de_streaming_transformer_l', type=str)
    parser.add_argument('-d', '--device', dest='device', default='cpu', help="Computation device. Either 'cpu' or 'cuda'. Note: Mac M1 / mps support isn't available yet.")    
    parser.add_argument('--lang', dest='language', default='', help='Explicity set language, default is empty = deduct languagefrom model tag', type=str)
    parser.add_argument('-b','--beamsize', dest='beamsize', help='Beam size for the decoder', type=int, default=5)
    parser.add_argument('--quiet', dest='quiet', help='No partial transcription output when transcribing a media file', action='store_true')
    parser.add_argument('--progress', dest='progress', help='Show progress when transcribing a media file', action='store_true')
    parser.add_argument('--save-debug-wav', dest='save_debug_wav', help='Save recording to debug.wav, only applicable to live decoding', action='store_true')
    parser.add_argument('--num-threads', dest='num_threads', default=1, help='Set number of threads used for intraop parallelism on CPU in pytorch.', type=int)

    parser.add_argument('inputfile', nargs='?', help='Input media file', default='')

    args = parser.parse_args()

    if args.num_threads != -1:
        torch.set_num_threads(args.num_threads)

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
