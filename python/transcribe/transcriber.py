from python.transcribe.utils import ffmpeg_microphone_live
import sys
import copy
import numpy as np
from transformers import pipeline
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

        self.run_checks()
        
    def run_checks(self):
        if int(self.chunk_length_s) != self.chunk_length_s:
            raise ValueError(
                    "Chunk length must be whole number."
                )
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
        
    def create_new_item(
            self,
            sampling_rate, 
            chunk=b"", 
            partial=False
        ):
        return {"raw": chunk, 
                "sampling_rate": sampling_rate, 
                "stride": (self.default_stride, self.default_stride), 
                "partial": partial}

    def transcribe(
            self,
            chunk_queue, 
            res_queues
        ):
        sampling_rate = self.transcriber.feature_extractor.sampling_rate

        mic_count = [0 for i in range(len(self.chunk_length_multipliers))]
        mic_list =[self.create_new_item(sampling_rate) for i in range(len(self.chunk_length_multipliers))]

        mic = ffmpeg_microphone_live(
            sampling_rate = sampling_rate,
            chunk_length_s = self.chunk_length_s,
            stream_chunk_s = self.stream_chunk_s,
            stride_length_s=0.0
        )

        print("Start speaking...")

        for item in mic:
            chunk_queue.put_nowait(item)
            for i in range(len(mic_count)):
                if mic_count[i] == 0:
                    mic_list[i] = copy.deepcopy(item)
                else:
                    mic_list[i]["raw"] += copy.deepcopy(item["raw"])

                mic_count[i] += self.chunk_length_s

                if mic_count[i] % (self.chunk_length_multipliers[i] * self.chunk_length_s) == 0:
                    mic_list[i]["raw"] = np.frombuffer(mic_list[i]["raw"], dtype=np.float32)
                    transcribed = self.transcriber(mic_list[i], generate_kwargs={"max_new_tokens": 1000000})
                    next_text = transcribed["text"]
                
                    if i > 0:
                        item_count = int(self.chunk_length_multipliers[i] / self.chunk_length_multipliers[i - 1])
                        for j in range(item_count):
                            res_queues[i - 1].get()
                    res_queues[i].put(next_text)

                    mic_count[i] = 0
                    mic_list[i] = self.create_new_item(sampling_rate)
