from utils import ffmpeg_microphone_live
import sys
import copy
import numpy as np
from colorama import Fore, Style

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color_code(index, text):
    colors = [Fore.CYAN, Fore.GREEN, Fore.YELLOW]
    print(f"{colors[index]}{text}{Style.RESET_ALL}")

def create_new_item(sampling_rate, stride=(0.0, 0.0), chunk=b"", partial=False):
    return {"raw": chunk, "sampling_rate": sampling_rate, "stride": stride, "partial": partial}

def transcribe(transcriber, chunk_yield_multiplier_list, chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic_count = [0 for i in range(len(chunk_yield_multiplier_list))]
    mic_list =[create_new_item(sampling_rate) for i in range(len(chunk_yield_multiplier_list))]
    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
        stride_length_s=0.0
    )

    result = [[] for i in range(len(chunk_yield_multiplier_list))]

    print("Start speaking...")

    for item in mic:
        for i in range(len(mic_count)):
            if mic_count[i] == 0:
                mic_list[i] = copy.deepcopy(item)
            else:
                mic_list[i]["raw"] += copy.deepcopy(item["raw"])

            mic_count[i] += chunk_length_s

            if mic_count[i] % (chunk_yield_multiplier_list[i] * chunk_length_s) == 0:
                mic_list[i]["raw"] = np.frombuffer(mic_list[i]["raw"], dtype=np.float32)
                transcribed = transcriber(mic_list[i], generate_kwargs={"max_new_tokens": 1000000})
                next_text = transcribed["text"]
            
                if i > 0:
                    result[i - 1] = []
                result[i].append(next_text)

                mic_count[i] = 0
                mic_list[i] = create_new_item(sampling_rate)

    return result

def launch_fn(
    classifier,
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Listening for wake word...")
    for prediction in classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                return True