import os
import json
import folder_paths
from server import PromptServer
from aiohttp import web
from .utils import get_api_key, get_models

class ListModelsNode:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.base_path, "model_lists")
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_provider": (["ollama", "llamacpp", "kobold", "lmstudio", "textgen", 
                                 "groq", "gemini", "openai", "anthropic", "mistral", 
                                 "transformers", "xai", "deepseek"], {"default": "ollama"}),
                "base_ip": ("STRING", {"default": "localhost"}),
                "port": ("STRING", {"default": "11434"}),
                "external_api_key": ("STRING", {"default": ""}),
                "refresh": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_list",)
    FUNCTION = "list_models"
    CATEGORY = "ImpactFrames💥🎞️/IF_LLM"

    def list_models(self, llm_provider, base_ip, port, external_api_key="", refresh=False):
        try:
            # Use external API key if provided, otherwise try to get from environment
            api_key = external_api_key if external_api_key else get_api_key(f"{llm_provider.upper()}_API_KEY", llm_provider)
            
            # Get models for the selected provider
            models = get_models(llm_provider, base_ip, port, api_key)
            
            # Format the output
            output = f"\n=== Available Models for {llm_provider.upper()} ===\n\n"
            
            if models:
                for i, model in enumerate(models, 1):
                    output += f"{i}. {model}\n"
            else:
                output += "No models available or provider requires valid API key/connection\n"
            
            # Save output to file
            file_path = os.path.join(self.output_dir, f"{llm_provider}_models.txt")
            with open(file_path, 'w') as f:
                f.write(output)
            
            return (output,)
            
        except Exception as e:
            error_msg = f"Error fetching models for {llm_provider}: {str(e)}"
            return (error_msg,)

# Add node class mappings
NODE_CLASS_MAPPINGS = {
    "ListModelsNode": ListModelsNode
}

# Add node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ListModelsNode": "List Available Models📋"
}
