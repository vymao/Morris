#include "generation.h"
#include <cpr/cpr.h>
#include "main/lib/util/common.h"

namespace generation {
    VirtualTextGenerator::VirtualTextGenerator() {
        json config_json = parseJSON("/Users/victor/Desktop/Morris/cpp/main/lib/llm-generator/config.json");
        from_json(config_json["model_settings"], llm_settings);
    }

    void VirtualTextGenerator::from_json(const json& j, llm_params& p) {
        try {
            j.at("model").get_to(p.model);
            j.at("prompt").get_to(p.prompt);
            j.at("temperature").get_to(p.temperature);
            j.at("top_p").get_to(p.top_p);
            j.at("top_k").get_to(p.top_k);
            j.at("max_tokens").get_to(p.max_tokens);
            j.at("repetition_penalty").get_to(p.repetition_penalty);
        }
        catch(...) {
            std::cerr << "Missing value in LLM generation config. Using default value instead." << std::endl;
        }
    }

    std::string VirtualTextGenerator::queryVirtualLLM(std::string& prompt) {
        json llm_config = config_json["model_settings"];
        llm_config["prompt"] = prompt;
        auto config_string = to_string(llm_config);
        cpr::Response r = cpr::Post(
            cpr::Url{config_json["endpoint_url"]},
            cpr::Header{{"accept", "application/json"}},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Bearer{config_json["api_key"]},
            cpr::Body{config_string});

        return r.text;

    }
}