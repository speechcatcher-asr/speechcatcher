# Note: the implementation follows the espnet example note book released here: https://github.com/espnet/notebook/blob/master/espnet2_streaming_asr_demo.ipynb
# ( Apache-2.0 license )

import sys
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from espnet_model_zoo.downloader import ModelDownloader
import argparse
import numpy as np
import wave

tag = "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024"

d=ModelDownloader(".cache/espnet")

speech2text = Speech2TextStreaming(**d.download_and_unpack(tag), token_type=None, bpemodel=None,
    maxlenratio=0.0, minlenratio=0.0, beam_size=10, ctc_weight=0.3, lm_weight=0.0,
    penalty=0.0, nbest=1, device = "cpu", disable_repetition_detection=True,
    decoder_text_length_limit=0, encoded_feat_length_limit=0
)

prev_lines = 0
def progress_output(text):
    global prev_lines
    lines=['']
    for i in text:
        if len(lines[-1]) > 100:
            lines.append('')
        lines[-1] += i
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
    sim_chunk_length = 640 #400
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

recognize("test1.wav")
