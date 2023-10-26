from utils import ffmpeg_microphone_live
import sys

def transcribe(transcriber, chunk_yield_multiplier_list, chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic_count = [0 for i in range(len(chunk_yield_multiplier_list))]
    mic_list =[b"" for i in range(len(chunk_yield_multiplier_list))]
    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")

    for item in mic:
        for i in range(len(mic_count)):
            mic_count[i] += 1
            mic_list[i] += item
            if chunk_yield_multiplier_list[i] % mic_count[i]:
                transcribed = transcriber(mic_list[i], generate_kwargs={"max_new_tokens": 50 * chunk_yield_multiplier_list[i]})
                next_text = transcribed["text"]
                print(next_text)
                mic_count[i] = 0
                mic_list[i] = b""

    return item["text"]

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