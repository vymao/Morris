// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "main/lib/util/common_sdl.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>

#include "audio_stream.h"

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t) {
    int64_t sec = t/100;
    int64_t msec = t - sec*100;
    int64_t min = sec/60;
    sec = sec - min*60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int) min, (int) sec, (int) msec);

    return std::string(buf);
}

void createAndRunAudioStream(whisper_params& params, std::queue<std::shared_ptr<std::vector<float>>>& classifier_data_queue) {
    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*params.audio_sampling_rate;
    const int n_samples_len  = (1e-3*params.length_ms)*params.audio_sampling_rate;
    const int n_samples_keep = (1e-3*params.keep_ms  )*params.audio_sampling_rate;
    const int n_samples_30s  = (1e-3*5000.0         )*params.audio_sampling_rate;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio

    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, params.audio_sampling_rate)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        exit;
    }

    audio.resume();

    std::shared_ptr<std::vector<float>> pcmf32_old = std::make_shared<std::vector<float>>();
    std::shared_ptr<std::vector<float>> pcmf32_new = std::make_shared<std::vector<float>>(n_samples_30s, 0.0f);

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/params.audio_sampling_rate,
                float(n_samples_len )/params.audio_sampling_rate,
                float(n_samples_keep)/params.audio_sampling_rate,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    bool is_running = true;

    printf("[Start speaking]\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    // main audio loop
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // process new audio
        {
            std::shared_ptr<std::vector<float>> pcmf32 = std::make_shared<std::vector<float>>(n_samples_30s, 0.0f);
            while (true) {
                audio.get(params.step_ms, *pcmf32_new);

                if ((int) pcmf32_new->size() > 2*n_samples_step) {
                    std::cout << "WARNING: cannot process audio fast enough, dropping audio" << std::endl;
                    audio.clear();
                    continue;
                }

                if ((int) pcmf32_new->size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new->size();
            std::cout << "n_samples_new: " << n_samples_new << std::endl;
            std::cout << "n_samples_len: " << n_samples_len << std::endl;
            std::cout << "n_samples_keep: " << n_samples_keep << std::endl;

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = 0; //std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));
            std:: cout << "left: " << (int) pcmf32_old->size() << ", right: " << std::max(0, n_samples_keep + n_samples_len - n_samples_new) << std::endl;

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());
            std::cout << "n_samples_take: " << n_samples_take << std::endl;
            std::cout << "pcmf32: " << pcmf32->size() << std::endl;
            pcmf32->resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                (*pcmf32)[i] = (*pcmf32_old)[pcmf32_old->size() - n_samples_take + i];
            }

            memcpy(pcmf32->data() + n_samples_take, pcmf32_new->data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
            classifier_data_queue.push(pcmf32);
        }
    }

    std::cout << "Ending audio stream..." << std::endl;
    audio.pause();
}