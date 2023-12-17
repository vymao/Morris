#include <queue>
#include <vector>

#include "main/lib/util/common.h"

void createAndRunAudioStream(whisper_params& params, std::queue<std::shared_ptr<std::vector<float>>>& classifier_data_queue);