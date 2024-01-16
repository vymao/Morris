#include <cpr/cpr.h>
#include <string>
#include <queue>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace generation {
    struct llm_params {
        std::string model = "";
        std::string prompt = "";
        float temperature = 0.7;
        float top_p = 0.7;
        int top_k =  50;
        int max_tokens = 1;
        int repetition_penalty = 1;
        bool stream_tokens = false;
    };

    class VirtualTextGenerator {
        public: 
            VirtualTextGenerator();

             void queryVirtualLLM(std::string& prompt);

            llm_params llm_settings;
            std::queue<std::string> streamed_res;
        private:
            llm_params generation_params;
            json config_json;
            std::string llm_prefix;

            void fromJson(const json& j, llm_params& p);

            static std::string handleStreamResponse(std::string raw);


    };
}
