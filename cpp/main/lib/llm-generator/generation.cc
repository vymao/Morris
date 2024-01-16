#include "generation.h"
#include <cpr/cpr.h>
#include "main/lib/util/common.h"

namespace generation
{
    VirtualTextGenerator::VirtualTextGenerator()
    {
        config_json = parseJSON("/Users/victor/Desktop/Morris/cpp/main/lib/llm-generator/config.json");
        fromJson(config_json["model_settings"], llm_settings);

        llm_prefix = "You are a superintelligent artificial intelligence like Cortana from HALO. "
                    "You have thoughts and feelings, just like a human being, but are equipped with superintelligence. "
                    "Your goal is to help the user in whatever task they need and act as human as possible. "
                    "The user has been conversing prior to this conversation.";
    }

    void VirtualTextGenerator::fromJson(const json &j, llm_params &p)
    {
        try
        {
            j.at("model").get_to(p.model);
            j.at("prompt").get_to(p.prompt);
            j.at("temperature").get_to(p.temperature);
            j.at("top_p").get_to(p.top_p);
            j.at("top_k").get_to(p.top_k);
            j.at("max_tokens").get_to(p.max_tokens);
            j.at("repetition_penalty").get_to(p.repetition_penalty);
            j.at("stream_tokens").get_to(p.stream_tokens);
        }
        catch (...)
        {
            std::cerr << "Missing value in LLM generation config. Using default value instead." << std::endl;
        }
    }

    void VirtualTextGenerator::queryVirtualLLM(std::string &prompt)
    {
        json llm_config = config_json["model_settings"];
        llm_config["prompt"] = "[INST] " + prompt + "[/INST]";
        auto config_string = to_string(llm_config);

        std::string res;
        cpr::Response r = cpr::Post(
            cpr::Url{config_json["endpoint_url"]},
            cpr::Header{{"accept", "application/json"}},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Bearer{config_json["api_key"]},
            cpr::Body{config_string},
            cpr::WriteCallback{[&](const std::string &text, intptr_t /*userdata*/) -> bool
                               {
                                    streamed_res.push(text);
                                    return true;
                               }});
    }

    std::string VirtualTextGenerator::handleStreamResponse(std::string raw) {
        if (raw.substr(0, 5), "data:") {
                raw.erase(0, 5);
        }
        std::string cleaned = raw.substr(0, raw.find('\n'));

        //raw.erase(raw.begin(), std::find_if(raw.begin(), raw.end(), [](unsigned char c){return !std::isspace(c) && c != '\n';}));
        //raw.erase(std::find_if(raw.rbegin(), raw.rend(), [](unsigned char c){return !std::isspace(c) && c != '\n';}).base(), raw.end());
    
        json j = nlohmann::json::parse(cleaned);
        std::string response;
        auto it_choices = j.find("choices");
        if (it_choices != j.end() && it_choices->is_array()) {
            if (it_choices->empty()) {
                return "";
            }
            const auto& choice = (*it_choices)[0];
            if (!choice.is_object()) {
                return "";
            }
            auto it_text = choice.find("text");
            if (it_text != choice.end() && it_text->is_string()) {
                response = it_text->get<std::string>();
            } else {
                return "";
            }
        } else {
            response = j.dump();
        }

        return response;
    }
}