#!/usr/bin/env python3

import asyncio
import websockets
import argparse
import wave
import subprocess
import sys
import io

# Function to convert audio using ffmpeg on the fly
def convert_audio(input_file, sample_rate, channels, bit_depth):
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', str(sample_rate),
        '-ac', str(channels),
        '-f', 'wav',
        '-acodec', 'pcm_s16le',  # 16-bit signed little-endian
        '-'
    ]
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout

# Function to check if wav file matches the desired format
def is_wav_compatible(wav_file, sample_rate, channels, bit_depth):
    with wave.open(wav_file, 'rb') as wf:
        return (wf.getframerate() == sample_rate and
                wf.getnchannels() == channels and
                wf.getsampwidth() == bit_depth // 8)

# Function to process and send audio to ASR websocket server
async def process_audio(websocket, audio_stream, sample_rate, is_wave):
    await websocket.send(f'{{ "config" : {{ "sample_rate" : {sample_rate} }} }}')
    
    buffer_size = int(sample_rate * 0.2)  # 0.2 seconds of audio
    while True:
        if is_wave:
            data = audio_stream.readframes(buffer_size)  # For wave files
        else:
            data = audio_stream.read(buffer_size)  # For ffmpeg stream

        if len(data) == 0:
            break
        await websocket.send(data)
        print(await websocket.recv())

    await websocket.send('{"eof" : 1}')
    print(await websocket.recv())

async def run_test(uri, input_file, sample_rate, channels, bit_depth):
    async with websockets.connect(uri) as websocket:
        if input_file.endswith('.wav') and is_wav_compatible(input_file, sample_rate, channels, bit_depth):
            print(f"Sending {input_file} directly as it is already in the correct format.")
            with wave.open(input_file, 'rb') as wf:
                await process_audio(websocket, wf, sample_rate, is_wave=True)
        else:
            print(f"Converting {input_file} using ffmpeg before sending.")
            audio_stream = convert_audio(input_file, sample_rate, channels, bit_depth)
            await process_audio(websocket, io.BufferedReader(audio_stream), sample_rate, is_wave=False)

def main():
    parser = argparse.ArgumentParser(description="Speechcatcher's Vosk-API WebSocket ASR test client with audio conversion")
    parser.add_argument('input_file', help='Path to the input audio file')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port (default: 8765)')
    parser.add_argument('--host', default='localhost', help='WebSocket server host (default: localhost)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--channels', type=int, default=1, help='Number of audio channels (default: 1)')
    parser.add_argument('--bit-depth', type=int, default=16, help='Bit depth (default: 16)')

    args = parser.parse_args()

    uri = f'ws://{args.host}:{args.port}'
    asyncio.run(run_test(uri, args.input_file, args.sample_rate, args.channels, args.bit_depth))

if __name__ == "__main__":
    main()

