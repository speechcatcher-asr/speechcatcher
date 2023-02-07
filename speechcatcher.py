# Note: the implementation follows the espnet example note book released here: https://github.com/espnet/notebook/blob/master/espnet2_streaming_asr_demo.ipynb
# The notebook ( Apache-2.0 license ) was released with the clear intention of sharing how to use streaming models with EspNet2

import sys
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from espnet_model_zoo.downloader import ModelDownloader
import argparse
import numpy as np
import wave
import pyaudio
import wavefile

tag = "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024"

d=ModelDownloader(".cache/espnet")

device = 'cpu'
#device = 'mps'

speech2text = Speech2TextStreaming(**d.download_and_unpack(tag), token_type=None, bpemodel=None,
    maxlenratio=0.0, minlenratio=0.0, beam_size=10, ctc_weight=0.3, lm_weight=0.0,
    penalty=0.0, nbest=1, device = device, disable_repetition_detection=True,
    decoder_text_length_limit=0, encoded_feat_length_limit=0
)

prev_lines = 0
def progress_output(text):
    global prev_lines
    lines=['']
    last_i=''
    for i in text:
        if len(lines[-1]) > 120:
            if last_i==' ' or last_i=='.' or last_i=='?' or last_i=='!': 
                lines.append('')
        lines[-1] += i
        last_i = i 
    for i,line in enumerate(lines):
        if i == prev_lines:
            sys.stderr.write('\n\r')
        else:
            sys.stderr.write('\r\033[B\033[K')
        sys.stderr.write(line)

    prev_lines = len(lines)
    sys.stderr.flush()

def recognize(wavfile):
    with wave.open(wavfile, 'rb') as wavfile:
        ch=wavfile.getnchannels()
        bits=wavfile.getsampwidth()
        rate=wavfile.getframerate()
        nframes=wavfile.getnframes()
        buf = wavfile.readframes(-1)
        data=np.frombuffer(buf, dtype='int16')
    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    sim_chunk_length = 640*4 #400
    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            results = speech2text(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                progress_output(nbests[0])
            else:
                progress_output("")
            
        results = speech2text(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
    else:
        results = speech2text(speech, is_final=True)
    nbests = [text for text, token, token_int, hyp in results]
    progress_output(nbests[0])

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
def recognize_microphone(record_max_seconds=120, channels=1, recording_format=pyaudio.paInt16,
                         samplerate=16000, chunksize=8192, save_debug_wav=False):
    list_microphones()
    blocks=[]

    stream = p.open(format=recording_format,channels=channels,rate=samplerate,input=True,frames_per_buffer=chunksize)
    print(f'Model {tag} fully loaded, starting live transcription with your microphone.')
    for i in range(0,int(samplerate/chunksize*record_max_seconds)+1):
        data=stream.read(chunksize)
        data=np.frombuffer(data, dtype='int16')
        if save_debug_wav:
            blocks.append(data)
        data=data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
        if i==int(samplerate/chunksize*record_max_seconds):
            results = speech2text(speech=data, is_final=True)
            break
        results = speech2text(speech=data, is_final=False)
        if results is not None and len(results) > 0:
            nbests = [text for text, token, token_int, hyp in results]
            text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
            progress_output(nbests[0])
        else:
            progress_output("")

    nbests = [text for text, token, token_int, hyp in results]
    progress_output(nbests[0])

    # Write debug wav as output file (will only be executed after shutdown)
    if save_debug_wav:
        print("Saving debug output...")
        wavefile.write("debug.wav", samplerate, np.concatenate(blocks, axis=None))
    else:
        print("Not writing debug wav output since --save_debug_wav is not set.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Speechcatcher utility to decode speech with speechcatcher espnet models.')
    parser.add_argument('-i', '--input-file', dest='input_file', help='Input wav file', type=str, default = 'test1.wav')
    parser.add_argument('-m', '--use-microphone', dest='microphone', help='Use microphone', type=int, default = 1)

    args = parser.parse_args()

    #recognize("test5.wav")
    recognize_microphone()
