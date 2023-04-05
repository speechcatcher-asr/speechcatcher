# Speechcatcher command line interface. Decode speech with speechcatcher streaming models. This tool can decode 
# either media files (any fileformat that ffmpeg supports) or live speech from a microphone.
#
# Note: the decoding implementation is inspired by the espnet example notebook released here:
# https://github.com/espnet/notebook/blob/master/espnet2_streaming_asr_demo.ipynb
#
# However, the implementation here is rewritten and substantially different
# Among other things, there is endpointing for the live and batch decoder. Threaded I/O for the microphone.
# Multi-processing for long audio files.
#
# The notebook ( Apache-2.0 license ) was released with the clear intention of sharing how to use
# streaming models with EspNet2.
#

import os
import sys
import argparse
import hashlib
#from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from speechcatcher.asr_inference_streaming import Speech2TextStreaming
from espnet_model_zoo.downloader import ModelDownloader
import numpy as np
import wave
import pyaudio
from scipy.io import wavfile
import ffmpeg
import torch
from tqdm import tqdm
import math
import json
import multiprocessing
import concurrent
import threading
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from speechcatcher.simple_endpointing import segment_speech

pbar_queue = None
speech2text_global = None
speech_global = None

tags = {
    "de_streaming_transformer_m": "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024",
    "de_streaming_transformer_l": "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_l_raw_de_bpe1024"}

# See https://stackoverflow.com/questions/75193175/why-i-cant-use-multiprocessing-queue-with-processpoolexecutor
def init_pool_processes(q, speech2text, speech):
    global pbar_queue
    global speech2text_global
    global speech_global

    pbar_queue = q
    speech2text_global = speech2text
    speech_global = speech


# Ensure that the directory for the path f exists
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


# Load the espnet model with the given tag
def load_model(tag, device='cpu', beam_size=5, quiet=False):
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

def progress_bar_output(q, max_i):
    pbar = tqdm(total=max_i)
    progress_i = 0
    while progress_i < max_i:
        q.get(block=True)
        progress_i += 1
        pbar.update(1)    

# Output current hypothesis on the fly. Note that .
def progress_output(text, prev_lines=0):
    lines = ['']
    last_i = ''
    for i in text:
        if len(lines[-1]) > 100:
            # make sure that we don'T
            if last_i == ' ' or last_i == '.' or last_i == '?' or last_i == '!':
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
def recognize_file(speech2text, media_path, output_file='', quiet=True, progress=True, num_processes=4):
    ensure_dir('.tmp/')
    wavfile_path = '.tmp/' + hashlib.sha1(args.inputfile.encode("utf-8")).hexdigest() + '.wav'
    convert_inputfile(media_path, wavfile_path)

    with wave.open(wavfile_path, 'rb') as wavfile_in:
        ch = wavfile_in.getnchannels()
        bits = wavfile_in.getsampwidth()
        rate = wavfile_in.getframerate()
        nframes = wavfile_in.getnframes()
        buf = wavfile_in.readframes(-1)
        raw_speech_data = np.frombuffer(buf, dtype='int16')

    chunk_length = 8192
    complete_text, paragraphs, paragraphs_tokens, paragraph_hyps, segments_start_end_in_seconds = recognize(speech2text, raw_speech_data, rate, chunk_length, num_processes, progress, quiet)

    # Automatically generate output .txt name from media_path if it isnt set
    # media_path can also be an URL, in that case it needs special handling
    if output_file == '':
        if media_path.startswith('http://') or media_path.startswith('https://'):
            output_file_txt = media_path.split('/')[-1] + '.txt'
            output_file_json = media_path.split('/')[-1] + '.json'
        else:
            output_file_txt = media_path + '.txt'
            output_file_json = media_path + '.json'
    else:
        output_file_txt = output_file + '.txt'
        output_file_json = output_file + '.json'

    with open(output_file_txt, 'w') as output_file_txt_out:
        output_file_txt_out.write(complete_text)

    with open(output_file_json, 'w') as output_file_json_out:
        complete_json = {"complete_text":complete_text, "segments_start_end_in_seconds":segments_start_end_in_seconds,
                        "segments":paragraphs, "segment_tokens":paragraphs_tokens}
        output_file_json_out.write(json.dumps(complete_json, indent=4))

    print(f'Wrote transcription to {output_file_txt} and {output_file_json}.')
    os.remove(wavfile_path)

# Recgonize the speech in 'raw_speech_data' with sampling rate 'rate' using the model in 'speech2text'.
# The rawspeech data should be a numpy array of dtype='int16'

def recognize(speech2text, raw_speech_data, rate, chunk_length=8192, num_processes=1, progress=True, quiet=False):
    # 32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    speech = raw_speech_data.astype(np.float16) / 32767.0

    speech_len = len(speech)
    speech_len_frames = (speech_len / rate) * 100.
    assert (rate == 16000)

    segments = []

    # segment audio files longer than a minute
    if speech_len > 60. * rate:
        segments = segment_speech(raw_speech_data, rate)

    # Get positions where we want to finalize. Everything here is still measured in frames (100 frames per sec).
    # Make sure the last segment is at least 10 seconds long, otherwise merge it with the previous one.
    segment_frame_pos_end = [segment[1] for segment in segments if segment[1] < speech_len_frames - 1000.]
    max_i = (speech_len // chunk_length) + 1

    segments_in_seconds = [0] + [f / 100. for f in segment_frame_pos_end] + [speech_len /rate]
    segments_in_seconds_start_end = list(zip(segments_in_seconds[:-1], segments_in_seconds[1:]))

    # The segments from segment_wav are in frame positions (100 frames per second)
    # with the given chunksize, we calculate the iterations here where we need to finalize
    # note: we do not finalize at the beginning, but at -1 to easily calculate start
    # and end positions for the loop below.
    segments_i = [-1] + [math.ceil((((f / 100.) * rate) - chunk_length) / chunk_length) for f in
                         segment_frame_pos_end] + [max_i]
    # print('finalize iterations:', segments_i)
    utterance_text = ''
    complete_text = ''
    paragraphs_raw = []
    speech_len = len(speech)
    seg_num = 1
    seg_num_total = len(segments_i) - 1
    futures = []
    q = multiprocessing.Queue()

    # Run the transcription of segments in parallel with multiple processes.
    # Note: we use a ProcessPoolExecutor to distribute the work.
    # The processes report their statuses back with a multiprocessing queue (interprocess communication).
    # The main process launches an additional thread to gather all status reports and displays a status bar with tqdm.
    # We initialize the multiprocessing queue, as well as the input speech and the speech2text module globally,
    # so that the spawned/forked processes can share and reuse them (otherwise it would be inefficiently pickled
    # as a function argument).
    if progress:
        t = threading.Thread(target=progress_bar_output, args=(q, max_i))
        t.start()
    with ProcessPoolExecutor(max_workers=num_processes, initializer=init_pool_processes,
                             initargs=(q, speech2text, speech)) as executor:

        start_end_positions = zip(segments_i[:-1], segments_i[1:])
        for start, end in start_end_positions:
            data_future = executor.submit(recognize_segment, speech_len, start, end, chunk_length,
                                          progress, rate, quiet)
            futures.append(data_future)

            seg_num += 1

        # wait until all segments have been recognized
        concurrent.futures.wait(futures)

        for r in futures:
            paragraphs_raw.append(r.result())

    paragraphs = [elem[0] for elem in paragraphs_raw] 

    paragraphs_tokens = [elem[1] for elem in paragraphs_raw]
    paragraphs_pos = [elem[2] for elem in paragraphs_raw]

    merged_paragraphs_tokens = list(itertools.chain(*[elem[1] for elem in paragraphs_raw]))
    merged_paragraphs_pos = list(itertools.chain(*[elem[2] for elem in paragraphs_raw]))

    paragraph_hyps = [elem[3] for elem in paragraphs_raw]

    assert(len(merged_paragraphs_tokens) == len(merged_paragraphs_pos))

    merged_paragraphs_json = zip(merged_paragraphs_tokens, merged_paragraphs_pos)
    merged_paragraphs = [paragraphs[0]] if len(paragraphs) > 0 else []

    # Post-processing for paragraphs
    # Merge paragraphs that shouldn't be separate segments
    # Make sure the first letter of a paragraph is upper case
    for prev_paragraph, paragraph in zip(paragraphs[:-1], paragraphs[1:]):
        # with endpointing, it's likely that there is a pause between the segments
        # here we actually check if the model thinks that this ending is also a sentence ending
        # only add a paragraph to the text output if model and end pointer agree on the segment boundary
        if is_completed(prev_paragraph):
            # Make sure the paragraph starts with a capitalized letter
            paragraph = upperCaseFirstLetter(paragraph)
            merged_paragraphs += [paragraph]
        else:
            # might be in the middle of a sentence - append to the last (open) paragraph
            merged_paragraphs[-1] += ' ' + paragraph
    complete_text = '\n\n'.join(merged_paragraphs)
    print('\n')
    return complete_text, paragraphs, paragraphs_tokens, paragraph_hyps, segments_in_seconds_start_end


# Transcribe a segment of speech, defined by start and end points
def recognize_segment(speech_len, start, end, chunk_length, progress, rate, quiet):
    segment_text = ''
    prev_lines = 0

    for i in range(start + 1, end + 1):
        speech_chunk_start = i * chunk_length
        speech_chunk_end = (i + 1) * chunk_length
        if speech_chunk_end > speech_len:
            speech_chunk_end = speech_len

        speech_chunk = speech_global[speech_chunk_start: speech_chunk_end]

        segment_text, prev_lines = batch_recognize_inner_loop(speech_chunk, i, prev_lines,
                                                              progress, quiet, rate, chunk_length,
                                                              is_final=(i == end))
        if progress:
            pbar_queue.put(1, block=False)
    return segment_text

# This advances the recognition by one step
def batch_recognize_inner_loop(speech_chunk, i, prev_lines, progress, quiet, rate,
                               chunk_length, is_final, debug_pos=False):

    if is_final and debug_pos:
       # first calculate in seconds, then multiply with 100 to get the framepos (100 frames in one second.)
       frame_pos = ((i + 1) * chunk_length / rate) * 100.
       print('is final @', f'{i}', f'{frame_pos}')

    results = speech2text_global(speech=speech_chunk, is_final=is_final, always_assemble_hyps= not (quiet or progress))
    utterance_text = ''
    utterance_token = []
    utterance_pos = []
    utterance_hyp = {}

    if quiet or progress:
        if is_final:
            #nbests = [text for text, token, token_int, hyp in results]
            
            utterance_text = results[0][0] if results is not None and len(results) > 0 else ""
            utterance_token = results[0][1] if results is not None and len(results) > 0 else []
            utterance_pos = results[0][-2] if results is not None and len(results) > 0 else []
            utterance_hyp = results[0][-3] if results is not None and len(results) > 0 else {}
    else:
        if results is not None and len(results) > 0:
            #nbests = [text, tokenpos for text, token, token_int, token_pos, hyp in results]
            utterance_text = results[0][0] #nbests[0] if nbests is not None and len(nbests) > 0 else ""
            utterance_token = results[0][1]
            utterance_pos = results[0][-2]
            utterance_hyp = results[0][-3]            

            prev_lines = progress_output(nbests[0], prev_lines)
        else:
            prev_lines = progress_output("", prev_lines)

    if is_final:
        prev_lines = 0

        if not (quiet or progress) and is_completed(utterance_text):
            sys.stdout.write('\n')

    return [utterance_text, utterance_token, utterance_pos, utterance_hyp], prev_lines


# List all available microphones on this system
def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    for i in range(0, num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print('Input Device id ', i, ' - ', p.get_device_info_by_host_api_device_index(0, i).get('name'))

# This gives the user a helpful error message and advices the user on possible solutions when we encounter an "input overflow".
def stream_read_with_exception(stream, chunksize, prev_lines, exception_on_overflow=True):
    try:
        data = stream.read(chunksize, exception_on_overflow=exception_on_overflow)
    except OSError as e:
        if 'Input overflowed' in str(e):
            print('\n')
            print('Input overflowed while trying to fetch new data from your microphone.')
            print('This happens when the online recognition was not fast enough to decode speech in realtime.')
            print('---')
            print('Solution 1: You can silently discard this error by running speechcatcher live transcription with the --no-exception-on-overflow option.')
            print('This may degrade recognition quality in unexpected ways, as some speech data will potentially be discarded to catch up with the newest microphone data.')
            print('or')
            print('Solution 2: Try to reduce the beamsize, for example with: speechcatcher -l -b 1. A smaller beamsize means faster decoding with slightly less accuracy.')
            print('and/or')
            print('Solution 3: Try to use a smaller and faster model.')
            print(prev_lines*'\n')
        else:
            # handle other types of OS errors
            print("An OS error occurred:", e)
        sys.exit(-1)
    return data

# Stream audio data from a microphone to an espnet model
# Chunksize should be at least 6400 for a lookahead of 16 frames
def recognize_microphone(speech2text, tag, record_max_seconds=120, channels=1, recording_format=pyaudio.paInt16,
                         samplerate=16000, chunksize=8192, save_debug_wav=False, exception_on_pyaudio_overflow=True,
                         finalize_update_iters=7):
    list_microphones()
    blocks = []

    p = pyaudio.PyAudio()
    stream = p.open(format=recording_format, channels=channels, rate=samplerate, input=True,
                    frames_per_buffer=chunksize)

    print(f'Model {tag} fully loaded, starting live transcription with your microphone.')

    n_best_lens = []
    prev_lines = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        data_future = executor.submit(stream_read_with_exception, stream, chunksize, prev_lines, exception_on_overflow=exception_on_pyaudio_overflow)
        for i in range(0, int(samplerate / chunksize * record_max_seconds) + 1):

            data = data_future.result(timeout=2)
            data_future = executor.submit(stream_read_with_exception, stream, chunksize, prev_lines, exception_on_overflow=exception_on_pyaudio_overflow)

            data = np.frombuffer(data, dtype='int16')
            if save_debug_wav:
                blocks.append(data)

            # 32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
            data = data.astype(np.float16) / 32767.0
            if i == int(samplerate / chunksize * record_max_seconds):
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
                #nbests = [text for text, token, token_int, hyp in results]
                nbests0 = results[0][0]
                nbest_len = len(nbests0)
                n_best_lens += [nbest_len]

                text = nbests0
                prev_lines = progress_output(text, prev_lines)
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


def main():
    parser = argparse.ArgumentParser(
        description='Speechcatcher utility to decode speech with speechcatcher espnet models.')
    parser.add_argument('-l', '--live-transcription', dest='live', help='Use microphone for live transcription',
                        action='store_true')
    parser.add_argument('-t', '--max-record-time', dest='max_record_time',
                        help='Maximum record time in seconds (live transcription).', type=float, default=120)
    parser.add_argument('-m', '--model', dest='model', default='de_streaming_transformer_l',
                        help='Choose a model: de_streaming_transformer_m or de_streaming_transformer_l', type=str)
    parser.add_argument('-d', '--device', dest='device', default='cpu',
                        help="Computation device. Either 'cpu' or 'cuda'."
                             " Note: Mac M1 / mps support isn't available yet.")
    parser.add_argument('--lang', dest='language', default='',
                        help='Explicitly set language, default is empty = deduct language from model tag', type=str)
    parser.add_argument('-b', '--beamsize', dest='beamsize', help='Beam size for the decoder', type=int, default=5)
    parser.add_argument('--quiet', dest='quiet', help='No partial transcription output when transcribing a media file',
                        action='store_true')
    parser.add_argument('--no-progress', dest='no_progress', help='Show no progress bar when transcribing a media file',
                        action='store_true')
    parser.add_argument('--no-exception-on-overflow', dest='no_exception_on_overflow',
                        help='Do not abort live recognition when encountering an input overflow.', action='store_true')
    parser.add_argument('--save-debug-wav', dest='save_debug_wav',
                        help='Save recording to debug.wav, only applicable to live decoding', action='store_true')
    parser.add_argument('--num-threads', dest='num_threads', default=1,
                        help='Set number of threads used for intraop parallelism on CPU in pytorch.', type=int)
    parser.add_argument('-n', '--num-processes', dest='num_processes', default=-1,
                        help='Set number of processes used for processing long audio files in parallel'
                             ' (the input file needs to be long enough). If set to -1, use multiprocessing.cpu_count() '
                             'divided by two.',
                        type=int)
    parser.add_argument('inputfile', nargs='?', help='Input media file', default='')

    args = parser.parse_args()

    if args.num_threads != -1:
        torch.set_num_threads(args.num_threads)

    num_processes = multiprocessing.cpu_count()

    # use n/2 as default for multi core machines with cores > 2 
    num_processes = num_processes // 2 if num_processes > 2 else num_processes

    if args.num_processes != -1:
        num_processes = args.num_processes

    if args.model not in tags:
        print(f'Model {args.model} is not a valid model!')
        print('Options are:', ', '.join(tags.keys()))
        sys.exit(-1)
    else:
        tag = tags[args.model]
        print('Using model: ', tag)
        if not args.live:
            print(f'Using {num_processes} processes for decoding. You can change this setting with the -n parameter option.')

    quiet = args.quiet or num_processes > 1
    progress = not args.no_progress

    speech2text = load_model(tag=tag, device=args.device, beam_size=args.beamsize, quiet=quiet or progress)

    args = parser.parse_args()

    if args.live:
        recognize_microphone(speech2text, tag, record_max_seconds=args.max_record_time,
                             save_debug_wav=args.save_debug_wav,
                             exception_on_pyaudio_overflow=not args.no_exception_on_overflow)
    elif args.inputfile != '':
        recognize_file(speech2text, args.inputfile, quiet=quiet, progress=progress, num_processes=num_processes)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
