from transformers import pipeline
import torch

from launch import launch_fn, transcribe

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
)

transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device=device
)

chunk_yield_minimum_length = 2.0
chunk_yield_length_multiples = [1.0, 5.0, 10.0]
#launch_fn(classifier, debug=True)
transcribe(transcriber, chunk_yield_length_multiples, chunk_length_s=chunk_yield_minimum_length, stream_chunk_s=3.0)