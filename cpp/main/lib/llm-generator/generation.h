#include <cpr/cpr.h>
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
    };

    class VirtualTextGenerator {
        public: 
            VirtualTextGenerator();

             std::string queryVirtualLLM(std::string& prompt);

            llm_params llm_settings;
        private:
            llm_params generation_params;
            json config_json;

            void from_json(const json& j, llm_params& p);


    };
}