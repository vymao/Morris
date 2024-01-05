# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH

class LibriSpeechDataset:
    def __init__(
        self,
        language: str = "english",
        task: str = "transcribe",
        generate_synthetic_labels: bool = False,
    ):
        config = AutoConfig.from_pretrained("openai/whisper-tiny.en")
        feature_extractor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        self.length = len(dataset)

        self.data = []
        self.labels = []

        for i in range(self.length):
            data = feature_extractor(dataset[i]['audio']['array'], sampling_rate = SAMPLE_RATE, return_tensors = 'pt')
            #if i % 100 == 0: print(data['input_values'].size())
            inputs = {
                "input_features": data['input_features'],
                "max_length": np.asarray([200], dtype=np.int32),
                "min_length": np.asarray([0], dtype=np.int32),
                "num_beams": np.asarray([1], dtype=np.int32),
                "num_return_sequences": np.asarray([1], dtype=np.int32),
                "length_penalty": np.asarray([1.0], dtype=np.float32),
                "repetition_penalty": np.asarray([1.0], dtype=np.float32),
                "early_stopping": np.asarray([True], dtype=np.bool_)
            }

            forced_decoder_ids = feature_extractor.get_decoder_prompt_ids(
                language=language, task=task, no_timestamps=True
            )
            forced_decoder_ids = [config.decoder_start_token_id, *[token[1] for token in forced_decoder_ids]]
            inputs["decoder_input_ids"] = np.asarray([forced_decoder_ids], dtype=np.int32)

            self.data.append(inputs)
            self.labels.append(dataset[i]['text'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1  # pylint: disable=unsubscriptable-object
        return data, label

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class LibriSpeechDataloader:
    def __init__(self, batch_size=1, generate_synthetic_labels = False,):
        self.batch_size = batch_size
        self.synthetic_labels = generate_synthetic_labels

        model_name = "openai/whisper-tiny.en"
        self.feature_extractor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        self.length = len(dataset)

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )

    def collate_batch(self, batch):
        input_padded = []
        labels = []

        for text in batch:
            data = self.feature_extractor(text['audio']['array'], sampling_rate = SAMPLE_RATE, return_tensors = 'pt')
            #if i % 100 == 0: print(data['input_values'].size())
            input_padded.append(data['input_features'])
            if not self.synthetic_labels: 
                labels.append(text['text'])
            else: 
                with torch.no_grad():
                    logits = self.model(data['input_features']).logits
                    predicted_class_id = torch.argmax(logits, dim=-1).item()
                    labels.append(predicted_class_id)

        return torch.vstack(input_padded), labels

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        try:
            for input, label in self.dataloader:
                yield input, label
        except StopIteration:
            return


def librispeech_dataloader(data_dir, batch_size=1, *args, **kwargs):
    return LibriSpeechDataset()

def librispeech_pt_dataloader(data_dir, batch_size, *args, **kwargs):
    return LibriSpeechDataloader(batch_size=batch_size)

def post_processing_func(output):
    #print("Res: ", torch.argmax(output, dim=-1))
    return torch.argmax(output, dim=-1)