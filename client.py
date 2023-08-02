import asyncio
import logging

import grpc
import proto.proto_pb2 as proto_pb2
import proto.proto_pb2_grpc as proto_pb2_grpc

import re

import requests

import sounddevice as sd
import wave
import numpy as np

def say_outloud(text):
    endpoint = "http://192.168.1.100:5500/api/tts"
    parameters = {
    "voice": "coqui-tts:en_vctk#p227",
    "text": text
    }
    response = requests.get(endpoint, parameters)
    if response.status_code == 200:
    # Save the audio file
    # print(response.content)
        with open('output.wav', 'wb') as audio_file:
            audio_file.write(response.content)
        wav_file = wave.open('output.wav', 'rb')
        
        # Get the properties of the audio file
        samplerate = wav_file.getframerate()
        data = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
        
        # Play the audio
        sd.play(data, samplerate)
        sd.wait()

def is_complete_sentence(sentence):
    if not sentence:
        return False
    if sentence[0].isupper() is False:
        return False
    if sentence[-1] not in '.!?':
        return False
    return True

async def run() -> None:
    conversation = ""
    full_answer = ""
    async with grpc.aio.insecure_channel("192.168.1.101:50051") as channel:
        stub = proto_pb2_grpc.LlamaCallbackStub(channel)

        # # Read from an async generator
        # async for response in stub.llamaAsk(proto_pb2.LlamaRequest(prompt="Hello")):
        #     print("Llama client received from async generator: "+ response.answer)

        # Direct read from the stub
        llama_stream = stub.llamaAsk(proto_pb2.LlamaRequest(prompt="What is 2 + 2?"))
        while True:
            response = await llama_stream.read()
            if response == grpc.aio.EOF:
                break
            # print("Llama client received from direct read: " + response.answer)
            conversation += response.answer
            full_answer += response.answer
            if is_complete_sentence(conversation.strip()):
                say_outloud(conversation.strip())
                conversation = ""
        print(full_answer)


if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(run())