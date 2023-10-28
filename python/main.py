from transformers import pipeline
import torch
import multiprocessing as mp
from ctypes import c_bool

from wake.wake_classifier import WakeClassifier
from transcribe.transcriber import Transcriber

from python.multiprocessing.queue import *


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier", "-c", type=str, default="MIT/ast-finetuned-speech-commands-v2"
    )
    parser.add_argument(
        "--transcriber", "-t", type=str, default="openai/whisper-base.en"
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    chunk_yield_minimum_length = 1.0
    chunk_yield_length_multiples = [2, 6, 12]

    wake = mp.Value(c_bool)
    chunk_queue = mp.Queue()

    res_queues = []
    for i in range(len(chunk_yield_length_multiples)):
        res_queues.append(CounterQueue())

    classifier = WakeClassifier(device=device)
    transcriber = Transcriber(
        chunk_yield_multiplier_list=chunk_yield_length_multiples,
        device=device,
        chunk_length_s=chunk_yield_minimum_length,
    )

    wake_process = mp.Process(
        name="wake_process_test_transcription",
        target=classifier.classify,
        args=(wake, chunk_queue, True),
    )

    transcribe_process = mp.Process(
        name="transcribe_process_test_transcription",
        target=transcriber.transcribe,
        args=(chunk_queue, res_queues),
    )
    wake_process.daemon = True
    transcribe_process.daemon = True

    wake_process.start()
    transcribe_process.start()

    while True:
        if wake.value:
            res = ""

            sizes = reversed([q.qsize() for q in res_queues])
            for q in res_queues[::-1]:
                try:
                    for j in range(next(sizes)):
                        res += q.get()
                except Exception as e:
                    print(e)
                    continue

            print(res)
            wake.value = False


if __name__ == "__main__":
    main()
