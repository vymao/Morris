# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from transformers import AutoFeatureExtractor, ASTForAudioClassification
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

class ASTDataset:
    def __init__(
        self,
        generate_synthetic_labels = False,
    ):
        model_name = "MIT/ast-finetuned-speech-commands-v2"
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-speech-commands-v2")
        dataset = load_dataset("speech_commands", "v0.02", split="test")
        if not generate_synthetic_labels:
            dataset = dataset.align_labels_with_mapping(model.config.label2id, "label")
        self.length = len(dataset)

        self.data = []
        self.labels = []

        for i in range(self.length):
            data = feature_extractor(dataset[i]['audio']['array'], sampling_rate = SAMPLE_RATE, return_tensors = 'pt')
            #if i % 100 == 0: print(data['input_values'].size())
            self.data.append(data['input_values'])
            if not generate_synthetic_labels: 
                self.labels.append(torch.tensor([dataset[i]['label']]))
            else: 
                with torch.no_grad():
                    logits = model(data['input_values']).logits
                    predicted_class_id = torch.argmax(logits, dim=-1).item()
                    self.labels.append(predicted_class_id)



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1  # pylint: disable=unsubscriptable-object
        return data, label

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ASTDataloader:
    def __init__(self, batch_size=1, generate_synthetic_labels = False,):
        self.batch_size = batch_size
        self.synthetic_labels = generate_synthetic_labels

        model_name = "MIT/ast-finetuned-speech-commands-v2"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-speech-commands-v2")

        dataset = load_dataset("speech_commands", "v0.02", split="test")
        if not generate_synthetic_labels:
            dataset = dataset.filter(lambda label: label != dataset.features["label"].str2int("_silence_"), input_columns="label")
            dataset = dataset.align_labels_with_mapping(self.model.config.label2id, "label")

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
            input_padded.append(data['input_values'])
            if not self.synthetic_labels: 
                labels.append(text['label'])
            else: 
                with torch.no_grad():
                    logits = self.model(data['input_values']).logits
                    predicted_class_id = torch.argmax(logits, dim=-1).item()
                    labels.append(predicted_class_id)

        return torch.vstack(input_padded), torch.Tensor(labels)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        try:
            for input, label in self.dataloader:
                yield input, label
        except StopIteration:
            return


def ast_audio_dataloader(data_dir, batch_size, *args, **kwargs):
    return ASTDataset()

def ast_audio_pt_dataloader(data_dir, batch_size, *args, **kwargs):
    return ASTDataloader(batch_size=batch_size, generate_synthetic_labels=False)

def post_processing_func(output):
    #print("Res: ", torch.argmax(output, dim=-1))
    return torch.argmax(output, dim=-1)