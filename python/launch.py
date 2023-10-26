from utils import ffmpeg_microphone_live
import sys

def transcribe(transcriber, chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    last_text = ""

    print("Start speaking...")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        #sys.stdout.write("\033[K")
        next_text = item["text"]
        """
        output_text = ""
        truncated_next_text = next_text[:len(last_text)]
        print("Truncated: ", truncated_next_text)
        print("Last: ", last_text)
        if last_text != truncated_next_text or not last_text:
            output_text = next_text
        else: 
            output_text = truncated_next_text

        last_text = next_text
        print(output_text)
        """
        print(next_text)
        #if not item["partial"][0]:
        #    break

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