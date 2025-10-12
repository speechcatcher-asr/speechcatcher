#!/usr/bin/env python3
"""Test our full transcription pipeline."""

from speechcatcher import speechcatcher
import numpy as np
import wave
import hashlib
import os

if __name__ == '__main__':
    print("="*80)
    print("OUR FULL PIPELINE TEST")
    print("="*80)

    # Load model
    print("\n[1] Loading model...")
    short_tag = 'de_streaming_transformer_xl'
    speech2text = speechcatcher.load_model(speechcatcher.tags[short_tag], beam_size=5, quiet=True)
    print("✅ Model loaded")

    # Load audio
    print("\n[2] Loading audio...")
    os.makedirs('.tmp/', exist_ok=True)
    wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
    if not os.path.exists(wavfile_path):
        speechcatcher.convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

    with wave.open(wavfile_path, 'rb') as wavfile_in:
        rate = wavfile_in.getframerate()
        buf = wavfile_in.readframes(-1)
        speech = np.frombuffer(buf, dtype='int16')

    print(f"Audio: rate={rate} Hz, shape={speech.shape}")

    # Transcribe
    print("\n[3] Running transcription...")
    complete_text, paragraphs = speechcatcher.recognize(speech2text, speech, rate, quiet=False, progress=True)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n✅ Complete text: '{complete_text}'")
    print(f"\n✅ Number of paragraphs: {len(paragraphs)}")

    if paragraphs:
        for i, para in enumerate(paragraphs):
            print(f"\nParagraph {i+1}:")
            print(f"  Text: '{para.get('text', '')}'")
            print(f"  Start: {para.get('start', 0):.2f}s, End: {para.get('end', 0):.2f}s")

            # Check token IDs if available
            if 'tokens' in para:
                tokens = para['tokens']
                print(f"  Tokens ({len(tokens)}): {tokens[:10]}...")

                # Check for problematic token 1023
                # Note: tokens might be strings not IDs in this output
                # Let's check the actual token representation
                if isinstance(tokens[0], str):
                    # Tokens are strings (BPE pieces)
                    if 'م' in tokens:
                        count = tokens.count('م')
                        print(f"  ⚠️  Arabic 'م' appears {count} times in tokens")
                    else:
                        print(f"  ✅ No Arabic characters in tokens")

    print("\n" + "="*80)
    print("OUR FULL PIPELINE TEST COMPLETE")
    print("="*80)
