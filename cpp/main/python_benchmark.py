from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
import time

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# audio file is decoded on the fly
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
print(inputs["input_values"].size())
with torch.no_grad():
    start = time.time()
    model(**inputs)
    end = time.time()
    print(end - start)
