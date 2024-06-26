# Speechcatcher streaming server interface. Decode speech with speechcatcher streaming models using live data from websockets.
#
# Note: work in progress, doesn't work correctly yet!

import asyncio
import websockets
import numpy as np
import argparse
import subprocess
import ffmpeg
from io import BytesIO
from threading import Thread
from queue import Queue, Empty
from speechcatcher.speechcatcher import (
    load_model, tags, progress_output, upperCaseFirstLetter,
    is_completed, ensure_dir, segment_speech, Speech2TextStreaming
)

class SpeechRecognitionSession:
    def __init__(self, speech2text, finalize_update_iters=7):
        self.speech2text = speech2text
        self.finalize_update_iters = finalize_update_iters
        self.n_best_lens = []
        self.prev_lines = 0
        self.blocks = []
        self.process = None
        self.stdout_queue = Queue()
        self.stderr_queue = Queue()
        self.start_ffmpeg_process()

    def start_ffmpeg_process(self):
        command = [
            "ffmpeg", "-loglevel", "debug", "-f", "webm", "-i", "pipe:0",
            "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "pipe:1"
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )
        Thread(target=self.read_from_ffmpeg_stdout, daemon=True).start()
        Thread(target=self.read_from_ffmpeg_stderr, daemon=True).start()

    def read_from_ffmpeg_stdout(self):
        while True:
            output = self.process.stdout.read(1024*2)  #(4096/4)
            if output:
                print("put output in queue:", len(output), type(output))
                self.stdout_queue.put(output)

    def read_from_ffmpeg_stderr(self):
        while True:
            error = self.process.stderr.readline()
            if error:
                print("FFmpeg stderr:", error.decode().strip())

    def decode_audio(self, audio_chunk):
        if self.process is None:
            self.start_ffmpeg_process()

        self.process.stdin.write(audio_chunk)
        self.process.stdin.flush()  # Ensure data is flushed through FFmpeg
        data = b''

        try:
            while not self.stdout_queue.empty():
                data += self.stdout_queue.get_nowait()
        except Empty:
            pass

        return np.frombuffer(data, dtype='int16')

    def process_audio_chunk(self, audio_chunk, is_final=False):
        print("Incoming audio data len:", len(audio_chunk), type(audio_chunk))

        data = self.decode_audio(audio_chunk)

        if data.size == 0:
            print("data was zero:", data)
            return ""

        data = data.astype(np.float16) / 32767.0  # Normalize

        print("data after decode audio is:", data)

        if is_final:
            results = self.speech2text(speech=data, is_final=True)
            nbests0 = results[0][0]
            return nbests0

        if len(self.n_best_lens) < self.finalize_update_iters:
            finalize_iteration = False
        else:
            if all(x == self.n_best_lens[-1] for x in self.n_best_lens[-10:]):
                finalize_iteration = True
                self.n_best_lens = []
            else:
                finalize_iteration = False

        results = self.speech2text(speech=data, is_final=finalize_iteration)

        print("results:", results)

        if results is not None and len(results) > 0:
            nbests0 = results[0][0]
            nbest_len = len(nbests0)
            self.n_best_lens.append(nbest_len)
            return nbests0
        return ""

async def recognize_ws(websocket, path, speech2text, format):
    print("Client connected")
    session = SpeechRecognitionSession(speech2text)
    try:
        async for message in websocket:
            transcription = session.process_audio_chunk(message, format)
            print("transcription:", transcription)
            if transcription:
                await websocket.send(str(transcription))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def start_server(host, port, speech2text, format):
    server = await websockets.serve(lambda ws, path: recognize_ws(ws, path, speech2text, format), host, port)
    await server.wait_closed()

def main():
    parser = argparse.ArgumentParser(description="Speechcatcher WebSocket Server for streaming ASR")
    parser.add_argument('--host', type=str, default='localhost', help='Host for the WebSocket server')
    parser.add_argument('--port', type=int, default=8765, help='Port for the WebSocket server')
    parser.add_argument('--model', type=str, default='de_streaming_transformer_xl', choices=tags.keys(),
                        help='Model to use for ASR')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Device to run the ASR model on ('cpu' or 'cuda')")
    parser.add_argument('--beamsize', type=int, default=5, help='Beam size for the decoder')
    parser.add_argument('--cache-dir', type=str, default='~/.cache/espnet', help='Directory for model cache')
    parser.add_argument('--format', type=str, default='pcm', choices=['pcm', 'webm', 'ogg'],
                        help='Audio format for the input stream')

    args = parser.parse_args()

    if args.model not in tags:
        print(f"Model {args.model} is not a valid model!")
        print("Options are:", ", ".join(tags.keys()))
        exit(1)

    tag = tags[args.model]
    print(f'Loading model: {tag}')

    speech2text = load_model(tag=tag, device=args.device, beam_size=args.beamsize, cache_dir=args.cache_dir)

    print(f'Starting WebSocket server on ws://{args.host}:{args.port}')
    asyncio.get_event_loop().run_until_complete(start_server(args.host, args.port, speech2text, args.format))
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    main()

