import argparse
import scipy
from scipy.io import wavfile
import ffmpeg
import math
from python_speech_features import logfbank
from scipy.ndimage import gaussian_filter1d
import numpy as np

"""
The endpointing algorithm below performs segmentation of long audio files by balancing segment length and energy-based decisions.
It aims to create segments of speech that are close to a specified average length, while also ensuring that cuts are made
at points where the energy of the audio signal is low (indicative of pauses or silences).

The algorithm uses beam search to explore different possible segmentations, optimizing for a combination of length reward
(how close the segment is to the target length) and energy at the cut (lower energy is preferable).

Weights can be applied to control the trade-off between these two factors:
- A higher length reward weight favors segments that are closer to the target length.
- A higher energy weight favors cuts made in lower-energy regions (such as pauses).
"""
class BeamSearch:
    def __init__(self, beam_size=10, ideal_segment_len=4000, max_lookahead=18000, min_len=2000, step=10,
                 len_reward_weight=1.0, energy_weight=1.0):
        self.beam_size = beam_size
        self.ideal_segment_len = ideal_segment_len
        self.max_lookahead = max_lookahead
        self.min_len = min_len
        self.step = step
        self.len_reward_weight = len_reward_weight
        self.energy_weight = energy_weight
        self.len_reward_factor = len_reward_weight / float(ideal_segment_len)

        self.length_rewards = []
        self.energy_at_cuts = []

    def cost_function(self, segment_length, energy_at_cut):
        length_reward = self.len_reward_factor * (self.ideal_segment_len - abs(self.ideal_segment_len - float(segment_length)))
        self.length_rewards.append(length_reward)
        self.energy_at_cuts.append(energy_at_cut)
        return (self.len_reward_weight * length_reward) + (self.energy_weight * energy_at_cut)

    def search(self, smoothed_energy, fbank_feat_len):
        sequences = [[[0], 0.0]]  # Start with the first position
        while True:
            all_candidates = []
            expand = False

            for seq_pos, current_score in sequences:
                last_cut = seq_pos[-1]
                score_at_k = sequences[-1][1]

                for j in range(self.min_len, min(self.max_lookahead, fbank_feat_len - last_cut - 1), self.step):
                    energy_at_cut = smoothed_energy[last_cut + j]
                    new_score = current_score + self.cost_function(j, energy_at_cut)

                    if new_score > current_score:
                        candidate = [seq_pos + [last_cut + j + 1], new_score]
                        all_candidates.append(candidate)
                    
                    if new_score > score_at_k:
                        expand = True

            if not all_candidates or not expand:
                break

            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:self.beam_size]

        best_cuts = sequences[0][0] if sequences[0][0] != [0] else [0, fbank_feat_len]
        return list(zip(best_cuts[:-1], best_cuts[1:]))

def segment_wav(wav_filename, beam_search, max_segment_len_sec=180, debug=False, visual_debug=False):
    samplerate, data = wavfile.read(wav_filename, mmap=False)
    return segment_speech(data, samplerate, beam_search, max_segment_len_sec, debug, visual_debug)

def segment_speech(data, samplerate, beam_search, max_segment_len_sec=180, debug=False, visual_debug=False):
    fbank_feat = logfbank(data, samplerate=samplerate, winlen=0.025, winstep=0.01)
    fbank_feat_power = fbank_feat.sum(axis=-1) / 10.0
    fbank_feat_power_smoothed = gaussian_filter1d(fbank_feat_power, sigma=20) * -1.0

    if visual_debug:
        plt.plot(fbank_feat_power_smoothed[:1000])
        plt.show()

    fbank_feat_len = len(fbank_feat)
    segments = beam_search.search(fbank_feat_power_smoothed, fbank_feat_len)

    # Ensure no segment is longer than max_segment_len_sec (in frames)
    max_segment_len_frames = max_segment_len_sec * 100  # 1 sec = 100 frames
    constrained_segments = []

    for start, end in segments:
        while end - start > max_segment_len_frames:
            constrained_segments.append((start, start + max_segment_len_frames))
            start += max_segment_len_frames
        constrained_segments.append((start, end))

    if debug:
        print_debug_info(beam_search, constrained_segments, samplerate)

    return constrained_segments

def print_debug_info(beam_search, segments, samplerate):
    print("\n--- Debug Info ---\n")
    
    # Segment statistics (in seconds)
    segment_lengths = [(end - start) / 100.0 for start, end in segments]
    mean_len = np.mean(segment_lengths)
    var_len = np.var(segment_lengths)
    min_len = np.min(segment_lengths)
    max_len = np.max(segment_lengths)
    
    print("Segment lengths (seconds):")
    for i, (start, end) in enumerate(segments):
        print(f"Segment {i + 1}: Start = {start / 100.0:.2f} sec, End = {end / 100.0:.2f} sec, Length = {(end - start) / 100.0:.2f} sec")

    print("\nSegment length statistics (seconds):")
    print(f"Mean: {mean_len:.2f}, Variance: {var_len:.2f}, Min: {min_len:.2f}, Max: {max_len:.2f}")

    # Length reward statistics
    length_rewards = beam_search.length_rewards
    mean_len_reward = np.mean(length_rewards)
    var_len_reward = np.var(length_rewards)
    min_len_reward = np.min(length_rewards)
    max_len_reward = np.max(length_rewards)

    print("\nLength reward statistics:")
    print(f"Mean: {mean_len_reward:.2f}, Variance: {var_len_reward:.2f}, Min: {min_len_reward:.2f}, Max: {max_len_reward:.2f}")

    # Energy at cut statistics
    energy_at_cuts = beam_search.energy_at_cuts
    mean_energy = np.mean(energy_at_cuts)
    var_energy = np.var(energy_at_cuts)
    min_energy = np.min(energy_at_cuts)
    max_energy = np.max(energy_at_cuts)

    print("\nEnergy at cut statistics:")
    print(f"Mean: {mean_energy:.2f}, Variance: {var_energy:.2f}, Min: {min_energy:.2f}, Max: {max_energy:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Endpointing tool to cut long audio into smaller pieces for ASR processing.')
    parser.add_argument('-a', '--average-segment-length', help='Average segment length in seconds.', type=float, default=60.0)
    parser.add_argument('-m', '--max-segment-length', help='Maximum segment length in seconds.', type=float, default=180.0)
    parser.add_argument('-b', '--beam-size', help='Beam size for beam search.', type=int, default=10)
    parser.add_argument('-s', '--step', help='Step size for beam search.', type=int, default=10)
    parser.add_argument('-r', '--len-reward', help='Length reward factor for beam search.', type=float, default=40)
    parser.add_argument('-lw', '--len-reward-weight', help='Weight for the length reward component.', type=float, default=12.0)
    parser.add_argument('-ew', '--energy-weight', help='Weight for the energy at cut component.', type=float, default=1.0)
    parser.add_argument('--debug', help='Enable debugging with detailed statistics.', action='store_true')
    parser.add_argument('--visual-debug', help='Enable visual debugging with plots.', action='store_true')
    parser.add_argument('filename', help='Path of the audio file.', type=str)
    args = parser.parse_args()

    if args.visual_debug:
        import matplotlib.pyplot as plt

    filename = args.filename
    tmp_file = f'tmp/{hex(abs(hash(filename)))[2:]}.wav'

    # Convert the input file to 16 kHz mono wav using FFmpeg
    (
        ffmpeg
        .input(filename)
        .output(tmp_file, acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(quiet=True)
    )

    beam_search = BeamSearch(
        beam_size=args.beam_size,
        ideal_segment_len=int(args.average_segment_length * 100),  # Convert seconds to frames (100 frames per second)
        step=args.step,
        len_reward_weight=args.len_reward_weight,
        energy_weight=args.energy_weight
    )

    result = segment_wav(tmp_file, beam_search, max_segment_len_sec=args.max_segment_length, 
                         debug=args.debug, visual_debug=args.visual_debug)
    print(result)

if __name__ == '__main__':
    main()

