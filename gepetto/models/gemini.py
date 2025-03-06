import google.genai as genai
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
    
    def query_model(self, query, cb, additional_model_options=None):
        if additional_model_options is None:
            additional_model_options = {}

        try:
            if type(query) == str:
                contents = query
            else:
                contents = []
                for history in query:
                    contents.append({
                        "parts":[{
                            "text": history['content']
                        }],
                        "role": history['role']
                    })
                if additional_model_options.get("stream"):
                    response = self.client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                    )
                    cb(response)
                    return
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    **{k: v for k, v in additional_model_options.items() if k != "stream"}
                )
                
                ida_kernwin.execute_sync(functools.partial(cb, response=response.text),
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