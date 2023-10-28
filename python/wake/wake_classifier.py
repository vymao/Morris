import numpy as np
from transformers import pipeline
import copy

class WakeClassifier:
    def __init__(
        self,
        wake_word="marvin",
        prob_threshold=0.5,
        classifier="MIT/ast-finetuned-speech-commands-v2",
        device="cpu",
    ):
        self.wake_word = wake_word
        self.prob_threshold = prob_threshold
        self.classifier = pipeline(
            "audio-classification", model=classifier, device=device
        )

        self.run_checks()
        
    def run_checks(self):
        if self.wake_word not in self.classifier.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {self.wake_word} not in set of valid class labels, pick a wake word in the set {self.classifier.model.config.label2id.keys()}."
            )
        
    def classify(
        self,
        wake_var, 
        chunk_queue,
        debug=False
    ):
        print("Listening for wake word...")
        while True:
            item = chunk_queue.get()
            cp = copy.deepcopy(item)
            prediction = self.classifier(np.frombuffer(cp["raw"], dtype=np.float32))[0]
            if debug:
                print(prediction)
            if prediction["label"] == self.wake_word:
                if prediction["score"] > self.prob_threshold:
                    wake_var.value = True