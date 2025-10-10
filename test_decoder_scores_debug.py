#!/usr/bin/env python3
"""Debug decoder scores to see why it predicts token 1023."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("DECODER SCORES DEBUG")
print("="*80)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile, load_model, tags

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    rate = wavfile_in.getframerate()
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

speech = raw_audio.astype(np.float32) / 32768.0

# Load model
print("\nLoading model...")
our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()
print("✅ Loaded\n")

# Process chunks until we have encoder output
chunk_size = 8000
for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    is_final = False

    with torch.no_grad():
        results = our_s2t(chunk, is_final=is_final)

    if our_s2t.beam_search.encoder_buffer is not None and our_s2t.beam_search.encoder_buffer.size(1) >= 40:
        print(f"Chunk {chunk_idx+1}: Encoder buffer has {our_s2t.beam_search.encoder_buffer.size(1)} frames")

        # Get encoder output (first 40 frames)
        encoder_out = our_s2t.beam_search.encoder_buffer.narrow(1, 0, 40)
        encoder_out_lens = torch.tensor([40], dtype=torch.long)

        print(f"\nTesting decoder with {encoder_out.shape} encoder output\n")

        # Initialize with SOS
        from speechcatcher.beam_search.hypothesis import create_initial_hypothesis
        hyp = create_initial_hypothesis(sos_id=1, device="cpu")

        print("="*60)
        print("STEP 1: Score all tokens from SOS")
        print("="*60)

        # Use beam search's batch_score_hypotheses (handles CTC + decoder)
        from speechcatcher.beam_search.beam_search import BeamSearch

        temp_beam_search = BeamSearch(
            scorers=our_s2t.beam_search.scorers,
            weights=our_s2t.beam_search.weights,
            beam_size=5,
            vocab_size=1024,
            sos_id=1,
            eos_id=2,
            device="cpu",
        )

        with torch.no_grad():
            combined_scores, _ = temp_beam_search.batch_score_hypotheses([hyp], encoder_out)

        # Show top 20 tokens
        top_scores, top_tokens = torch.topk(combined_scores[0], 20)

        dec_weight = our_s2t.beam_search.weights["decoder"]
        ctc_weight = our_s2t.beam_search.weights["ctc"]
        print(f"\nTop 20 tokens (combined = {dec_weight:.1f}*decoder + {ctc_weight:.1f}*ctc):")
        print(f"{'Rank':<6} {'Token':<8} {'Score':<12}")
        print("-" * 30)
        for i, (score, token) in enumerate(zip(top_scores.tolist(), top_tokens.tolist())):
            print(f"{i+1:<6} {token:<8} {score:>11.4f}")

        # Check specific tokens
        print(f"\nSpecific tokens:")
        print(f"  Token 1023 (م): {combined_scores[0, 1023].item():.4f}")

        # Load BPE to decode some tokens
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load("/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/data/de_token_list/bpe_unigram1024/bpe.model")

            print(f"\nTop tokens decoded:")
            for i, token in enumerate(top_tokens[:10].tolist()):
                piece = sp.IdToPiece(token)
                print(f"  {i+1}. Token {token}: '{piece}'")
        except:
            pass

        break

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
