import google.genai as genai
import google.genai.types as genai_types
import os
import functools
import threading
import ida_kernwin

import gepetto.config
import gepetto.models.model_manager
from gepetto.models.base import LanguageModel

GEMINI_FLASH_NAME = "gemini-2.0-flash-001"
GEMINI_FLASH_THINKING_NAME = "gemini-2.0-flash-thinking-exp"
GEMINI_PRO_NAME = "gemini-2.0-pro-exp-02-05"

safetySettings = [
    genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
    genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
    genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
    genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
    genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
    genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,threshold=genai_types.HarmBlockThreshold.BLOCK_NONE),
]

class Gemini(LanguageModel):
    @staticmethod
    def get_menu_name() -> str:
        return "GOOGLE-GEMINI"
    
    @staticmethod
    def supported_models():
        return [GEMINI_FLASH_NAME, GEMINI_FLASH_THINKING_NAME, GEMINI_PRO_NAME]
    
    @staticmethod
    def is_configured_properly() -> bool:
        return bool(gepetto.config.get_config("GoogleGemini", "API_KEY", "GOOGLE_GEMINI_API_KEY"))
    
    def __init__(self, model):
        self.model = model
        api_key = gepetto.config.get_config("GoogleGemini", "API_KEY", "GOOGLE_GEMINI_API_KEY")
        if not api_key:
            raise ValueError(_("Please edit the configuration file to insert your {api_provider} API key!")
                             .format(api_provider="GoogleGemini"))
        
        proxy = gepetto.config.get_config("Gepetto", "PROXY")
        if proxy and proxy.startswith("http"):
            os.environ['HTTP_PROXY'] = proxy
        
        self.client = genai.Client(api_key=api_key)

    def __str__(self):
        return self.model
    
    def _map_openai_config_to_gemini(self, openai_config):
        """
        将OpenAI风格的配置参数转换为Gemini API可用的配置参数
        """
        if not openai_config:
            return {}
            
        gemini_config = {}
        
        # 映射兼容的参数
        if "temperature" in openai_config:
            gemini_config["temperature"] = openai_config["temperature"]
            
        if "top_p" in openai_config:
            gemini_config["top_p"] = openai_config["top_p"]
            
        if "top_k" in openai_config:
            gemini_config["top_k"] = openai_config["top_k"]
            
        if "max_tokens" in openai_config:
            gemini_config["max_output_tokens"] = openai_config["max_tokens"]
            
        if "n" in openai_config:
            gemini_config["candidate_count"] = openai_config["n"]
            
        if "stop" in openai_config:
            gemini_config["stop_sequences"] = openai_config["stop"] if isinstance(openai_config["stop"], list) else [openai_config["stop"]]
            
        if "presence_penalty" in openai_config:
            gemini_config["presence_penalty"] = openai_config["presence_penalty"]
            
        if "frequency_penalty" in openai_config:
            gemini_config["frequency_penalty"] = openai_config["frequency_penalty"]
            
        if "logprobs" in openai_config:
            gemini_config["logprobs"] = openai_config["logprobs"]
            
        if "seed" in openai_config:
            gemini_config["seed"] = openai_config["seed"]
        
        return gemini_config
    
    def query_model(self, query, cb, additional_model_options=None):
        config = genai_types.GenerateContentConfig()
        config.safety_settings = safetySettings
        
        # 将OpenAI风格的配置转换为Gemini配置
        if additional_model_options:
            gemini_options = self._map_openai_config_to_gemini(additional_model_options)
            # 应用转换后的配置参数
            for key, value in gemini_options.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        try:
            if type(query) == str:
                contents = query
            else:
                contents = []
                for history in query:
                    if history['role'] == 'system':
                        config.system_instruction = history['content']
                        continue
                    contents.append({
                        "parts":[{
                            "text": history['content']
                        }],
                        "role": history['role']
                    })
                if additional_model_options and additional_model_options.get("stream"):
                    response = self.client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=config
                    )
                    cb(response)
                    return
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                
                ida_kernwin.execute_sync(functools.partial(cb, response=response.candidates[0].content.parts[0].text),
                                    ida_kernwin.MFF_WRITE)
        except genai.types.generation_types.StopCandidateException as e:
            print(_("Safety issue encountered: {error}").format(error=str(e)))
        except Exception as e:
            print(_("General exception encountered while running the query: {error}").format(error=str(e)))

    def query_model_async(self, query, cb, additional_model_options=None):
        """
        Function which sends a query to Gemini and calls a callback when the response is available.
        :param query: The request to send to Gemini
        :param cb: The function to which the response will be passed to.
        :param additional_model_options: Additional parameters used when creating the model object.
        """
        t = threading.Thread(target=self.query_model, args=[query, cb, additional_model_options])
        t.start()

gepetto.models.model_manager.register_model(Gemini)