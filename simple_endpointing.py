#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright 2022 HITeC e.V., Benjamin Milde and Robert Geislinger
#
#    Licensed under the Apache License, Version 2.0 (the 'License');
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an 'AS IS' BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import scipy
from scipy.io import wavfile
import ffmpeg
import pylab as plt
import math
from python_speech_features import logfbank
from scipy.ndimage import gaussian_filter1d
import audiosegment
import numpy as np

# All timing are in frames, where one frame is 0.01 seconds.
def process_wav(wav_filename, beam_size=10, ideal_segment_len=1000*4,
                max_lookahead=100*180, min_len=1000*2, step=10, len_reward = 40, debug=False):

    samplerate, data = wavfile.read(wav_filename, mmap=False)
    fbank_feat = logfbank(data, samplerate=samplerate, winlen=0.025, winstep=0.01)
    fbank_feat_power = fbank_feat.sum(axis=-1) / 10.
    
    fbank_feat_len = len(fbank_feat)

    fbank_feat_min_power = min(fbank_feat_power)
    fbank_feat_max_power = max(fbank_feat_power)

    fbank_feat_power_smoothed = gaussian_filter1d(fbank_feat_power, sigma=20) * -1.0
    
    if debug:
        print('min:', fbank_feat_min_power, 'max:', fbank_feat_max_power)

    if debug:
        plt.imshow(fbank_feat[:1000].T, interpolation=None, aspect='auto', origin='lower')
        plt.show()
        plt.plot(fbank_feat_power_smoothed[:1000])
        plt.show()

    cont_search = True

    len_reward_factor = len_reward / float(ideal_segment_len)

    # Simple Beam search to find good cuts, where the eneregy is low and where its
    # still close to the ideal segment length.
    # Sequences are of this shape; first list keeps track of the split positions,
    # the float value is the combined score for the complete path.
    sequences = [[[0], 0.0]]
    sequences_ordered = [[]]
    
    while cont_search:
        all_candidates = sequences
        cont_search = False
        # Expand each current candidate
        for i in range(len(sequences)):
            seq_pos, current_score = sequences[i]
            last_cut = (seq_pos[-1] if (len(seq_pos) > 0) else 0)
            score_at_k = sequences[-1][1]
            # Search over all tokens, min_len to max_lookahead
            for j in range(min_len, min(max_lookahead, fbank_feat_len - last_cut - 1), step): # <-- TODO without -1?
                len_reward = len_reward_factor * (ideal_segment_len - math.fabs(ideal_segment_len - float(j)))
                fbank_score = fbank_feat_power_smoothed[last_cut+j]
                new_score = current_score + len_reward + fbank_score
                if new_score > current_score:
                    candidate = [seq_pos + [last_cut + j + 1], new_score]
                    all_candidates.append(candidate)
                # Only continue the search, of at least one of the candidates was better than the current score at k
                if new_score > score_at_k:
                    cont_search = True

        # Order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        # Select k best
        sequences_ordered = ordered[:beam_size]
        sequences = sequences_ordered

    # This can happen with very short input wavs
    if len(sequences_ordered[0][0]) <= 1:
        segments = [(0, fbank_feat_len)]
    else:
        best_cuts = sequences_ordered[0]
        segments = list(zip(best_cuts[0][:-1], best_cuts[0][1:]))
    
    # This prevents the overlapping of segments
    segments = [(x[0]+1, x[1]) if x[0]!=0 else (x[0], x[1]) for x in segments]
    
    # Write wave segments
    filenameS = wav_filename.rpartition('.')[0] # Filename without file extension
    filenameRS = filenameS.partition('/')[2]
    
    text = ""
    count = 0
    segmentsFN = f'{filenameS}_segments'
    for a in segments:
        start = a[0]/100
        end = a[1]/100
        count_str = "%.4d" % count
        text += f'{filenameRS}_{count_str} {filenameRS} {start} {end}\n'
        count+=1
    with open(segmentsFN, 'w') as f:
        f.write(text)
    return segmentsFN, segments
    
    
    filename_list = []
    segment_count = 0
    silence = audiosegment.silent(duration=220, frame_rate=samplerate)
    silenceS = audiosegment.silent(duration=115, frame_rate=samplerate)
    for i, segment in enumerate(segments):
        actual_segment = silence.to_numpy_array()
        actual_segment = np.append(actual_segment, data[segment[0]*160:segment[1]*160])
        actual_segment = np.append(actual_segment, silenceS.to_numpy_array())
        out_filename = f'{filenameS}_{i}.wav'
        wavfile.write(out_filename, samplerate, actual_segment)
        segment_count = i
        filename_list.append(out_filename)
    
    last_segment = silence.to_numpy_array()
    last_segment = np.append(last_segment, data[segments[-1][1]*160:])
    if segments[-1][1] < fbank_feat_len:
        filename_last_segment = f'{filenameS}_{segment_count + 1}.wav'
        wavfile.write(filename_last_segment, samplerate, last_segment)
        filename_list.append(filename_last_segment)
        segments.append((segments[-1][1]+1, fbank_feat_len))

    return filename_list, segments


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='This tool does a simple endpointing beam search over a long audio'
                                                 ' file, to cut it into smaller pieces for ASR processing.')

    parser.add_argument('-a', '--average-segment-length', help='Average segment length in seconds.',
                                     type=float, default=60.0)

    # Positional argument, without (- and --)
    parser.add_argument('filename', help='The path of the mediafile', type=str)

    args = parser.parse_args()
    filenameS = args.filename.rpartition('.')[0] # Filename without file extension
    filename = args.filename

    filenameS_hash = hex(abs(hash(filenameS)))[2:]

    tmp_file = f'tmp/{filenameS_hash}.wav'

    # Use FFmpeg to convert the input media file to 16 kHz wav mono
    (
        ffmpeg
            .input(filename)
            .output(tmp_file, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
    )

    result = process_wav(tmp_file, debug=False)
