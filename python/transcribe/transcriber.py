from python.transcribe.utils import ffmpeg_microphone_live
import sys
import copy
import numpy as np
from transformers import pipeline
import multiprocessing as mp
from python.multiprocessing.queue import *


class Transcriber:
    def __init__(
        self,
        chunk_yield_multiplier_list=[2, 6, 12],
        transcriber="openai/whisper-base.en",
        chunk_length_s=5.0,
        stream_chunk_s=1.0,
        default_stride=0.0,
        device="cpu",
    ):
        self.chunk_length_s = chunk_length_s
        self.chunk_length_multipliers = chunk_yield_multiplier_list
        self.stream_chunk_s = stream_chunk_s
        self.default_stride = default_stride
        self.transcriber = pipeline(
            "automatic-speech-recognition", model=transcriber, device=device
        )

        self._run_checks()

    def _run_checks(self):
        if int(self.chunk_length_s) != self.chunk_length_s:
            raise ValueError("Chunk length must be whole number.")
        for value in self.chunk_length_multipliers:
            if not isinstance(value, int):
                raise ValueError(
                    "All chunk length multipliers must be whole numbers and ints."
                )

        for i in range(len(self.chunk_length_multipliers) - 1):
            if self.chunk_length_multipliers[i + 1] % self.chunk_length_multipliers[i]:
                raise ValueError(
                    "Sequential chunk length multipliers must be divisible."
                )

    def _process_audio(
        self,
        item_queues,
        res_queues,
        idx,
        chunk_length_multipliers,
        min_chunk_length_s,
        sampling_rate,
    ):
        item = self.create_new_item(sampling_rate)
        mic_count = 0

        while True:
            next_item = item_queues[idx].get()
            if mic_count == 0:
                item = copy.deepcopy(next_item)
            else:
                item["raw"] += copy.deepcopy(next_item["raw"])

            mic_count += min_chunk_length_s

            if mic_count % (chunk_length_multipliers[idx] * min_chunk_length_s) == 0:
                item["raw"] = np.frombuffer(item["raw"], dtype=np.float32)
                transcribed = self.transcriber(
                    item, generate_kwargs={"max_new_tokens": 1000000}
                )
                next_text = transcribed["text"]

                if idx > 0:
                    item_count = int(
                        self.chunk_length_multipliers[idx]
                        / self.chunk_length_multipliers[idx - 1]
                    )
                    for j in range(item_count):
                        res_queues[idx - 1].get()
                res_queues[idx].put(next_text)

                mic_count = 0
                item = self.create_new_item(sampling_rate)

    def create_processes(
        self,
        item_queues,
        res_queues,
        sampling_rate,
    ):
        print("Starting transcription processes...", end="")
        for i in range(len(self.chunk_length_multipliers)):
            transcribe_process = mp.Process(
                name=f"transcribe_process_{i}",
                target=self._process_audio,
                args=(
                    item_queues,
                    res_queues,
                    i,
                    self.chunk_length_multipliers,
                    self.stream_chunk_s,
                    sampling_rate,
                ),
            )

            transcribe_process.daemon = True
            transcribe_process.start()

        sys.stdout.flush()
        print("\rStarting transcription processes...done.")

    def create_new_item(self, sampling_rate, chunk=b"", partial=False):
        return {
            "raw": chunk,
            "sampling_rate": sampling_rate,
            "stride": (self.default_stride, self.default_stride),
            "partial": partial,
        }

    def transcribe(self, classify_queue, item_queues):
        sampling_rate = self.transcriber.feature_extractor.sampling_rate

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=self.chunk_length_s,
            stream_chunk_s=self.stream_chunk_s,
            stride_length_s=0.0,
        )

        print("Start speaking...")

        for item in mic:
            for queue in item_queues:
                queue.put(item)

            classify_queue.put_nowait(item)
