# IFPROMPTImaGENNode.py
import os
import sys
import json
import torch
import asyncio
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Tuple
import folder_paths
from .omost import omost_function
from .send_request import send_request
from .utils import (
    get_api_key,
    get_models,
    process_images_for_comfy,
    clean_text,
    load_placeholder_image,
    validate_models,
    save_combo_settings,
    load_combo_settings,                            
    create_settings_from_ui,
    prepare_batch_images,
    process_auto_mode_images
)
import base64
import numpy as np

# Add ComfyUI directory to path
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, comfy_path)

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/IF_PROMPTImaGEN/get_llm_models")
    async def get_llm_models_endpoint(request):
        try:
            data = await request.json()
            llm_provider = data.get("llm_provider")
            engine = llm_provider
            base_ip = data.get("base_ip")
            port = data.get("port")
            external_api_key = data.get("external_api_key")
        
            if external_api_key:
                api_key = external_api_key
            else:
                api_key_name = f"{llm_provider.upper()}_API_KEY"
                try:
                    api_key = get_api_key(api_key_name, engine)
                except ValueError:
                    api_key = None

            node = IFPROMPTImaGEN()
            models = node.get_models(engine, base_ip, port, api_key)
            return web.json_response(models)
        
        except Exception as e:
            print(f"Error in get_llm_models_endpoint: {str(e)}")
            return web.json_response([], status=500)

    @PromptServer.instance.routes.post("/IF_PROMPTImaGEN/add_routes")
    async def add_routes_endpoint(request):
        return web.json_response({"status": "success"})

    @PromptServer.instance.routes.post("/IF_PROMPTImaGEN/save_combo_settings")
    async def save_combo_settings_endpoint(request):
        try:
            data = await request.json()
            
            # Convert UI settings to proper format
            settings = create_settings_from_ui(data)
            
            # Get node instance
            node = IFPROMPTImaGEN()
            
            # Save settings
            saved_settings = save_combo_settings(settings, node.combo_presets_dir)
            
            return web.json_response({
                "status": "success",
                "message": "Combo settings saved successfully",
                "settings": saved_settings
            })
            
        except Exception as e:
            logger.error(f"Error saving combo settings: {str(e)}")
            return web.json_response({
                "status": "error", 
                "message": str(e)
            }, status=500)

except AttributeError:
    print("PromptServer.instance not available. Skipping route decoration for IF_PROMPTImaGEN.")

class IFPROMPTImaGEN:
    def __init__(self):
        self.strategies = "normal"
        # Initialize paths and load presets
        # self.base_path = folder_paths.base_path
        self.presets_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_PromptImaGen", "IF_AI", "presets")
        self.combo_presets_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_PromptImaGen", "IF_AI", "presets", "AutoCombo")
        # Load preset configurations
        self.profiles = self.load_presets(os.path.join(self.presets_dir, "profiles.json"))
        self.neg_prompts = self.load_presets(os.path.join(self.presets_dir, "neg_prompts.json"))
        self.embellish_prompts = self.load_presets(os.path.join(self.presets_dir, "embellishments.json"))
        self.style_prompts = self.load_presets(os.path.join(self.presets_dir, "style_prompts.json"))
        self.stop_strings = self.load_presets(os.path.join(self.presets_dir, "stop_strings.json"))

        # Initialize placeholder image path
        self.placeholder_image_path = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_PromptImaGen", "IF_AI", "placeholder.png")

        # Default values

        self.base_ip = "localhost"
        self.port = "11434"
        self.engine = "ollama"
        self.selected_model = ""
        self.profile = "IF_PromptMKR_IMG"
        self.messages = []
        self.keep_alive = False
        self.seed = 94687328150
        self.history_steps = 10
        self.external_api_key = ""
        self.preset = "Default"
        self.precision = "fp16"
        self.attention = "sdpa"
        self.Omni = None
        self.mask = None
        self.aspect_ratio = "1:1"
        self.keep_alive = False
        self.clear_history = False
        self.random = False
        self.max_tokens = 2048
        self.temperature = 0.8
        self.top_k = 40
        self.top_p = 0.9
        self.repeat_penalty = 1.1
        self.batch_count = 4

    @classmethod
    def INPUT_TYPES(cls):
        node = cls() 
        return {
            "required": {
                "images": ("IMAGE", {"list": True}),  # Primary image input
                "llm_provider": (["xai","llamacpp", "ollama", "kobold", "lmstudio", "textgen", "groq", "gemini", "openai", "anthropic", "mistral", "transformers"], {}),
                "llm_model": ((), {}),
                "base_ip": ("STRING", {"default": "localhost"}),
                "port": ("STRING", {"default": "11434"}),
                "user_prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "strategy": (["normal", "omost", "create", "edit", "variations"], {"default": "normal"}),
                "mask": ("MASK", {}),
                "prime_directives": ("STRING", {"forceInput": True, "tooltip": "The system prompt for the LLM."}),
                "profiles": (["None"] + list(cls().profiles.keys()), {"default": "None", "tooltip": "The pre-defined system_prompt from the json profile file on the presets folder you can edit or make your own will be listed here."}),
                "embellish_prompt": (list(cls().embellish_prompts.keys()), {"tooltip": "The pre-defined embellishment from the json embellishments file on the presets folder you can edit or make your own will be listed here."}),
                "style_prompt": (list(cls().style_prompts.keys()), {"tooltip": "The pre-defined style from the json style_prompts file on the presets folder you can edit or make your own will be listed here."}),
                "neg_prompt": (list(cls().neg_prompts.keys()), {"tooltip": "The pre-defined negative prompt from the json neg_prompts file on the presets folder you can edit or make your own will be listed here."}),
                "stop_string": (list(cls().stop_strings.keys()), {"tooltip": "Specifies a string at which text generation should stop."}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "tooltip": "Maximum number of tokens to generate in the response."}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature", "tooltip": "Toggles between using a fixed seed or temperature-based randomness."}),
                "seed": ("INT", {"default": 0, "tooltip": "Random seed for reproducible outputs."}),
                "keep_alive": ("BOOLEAN", {"default": False, "label_on": "Keeps Model on Memory", "label_off": "Unloads Model from Memory", "tooltip": "Determines whether to keep the model loaded in memory between calls."}),
                "clear_history": ("BOOLEAN", {"default": True, "label_on": "Clear History", "label_off": "Keep History", "tooltip": "Determines whether to clear the history between calls."}),
                "history_steps": ("INT", {"default": 10, "tooltip": "Number of steps to keep in history."}),
                "aspect_ratio": (["1:1", "16:9", "4:5", "3:4", "5:4", "9:16"], {"default": "1:1", "tooltip": "Aspect ratio for the generated images."}),
                "auto": ("BOOLEAN", {"default": False, "label_on": "Auto Is Enabled", "label_off": "Auto is Disabled", "tooltip": "If true, it generates auto promts based on the listed images click the save combomix settings to set the auto prompt generation file"}),
                "auto_mode": ("BOOLEAN", {"default": False, "label_on": "Auto Mix", "label_off": "Auto Combo", "tooltip": "If true, it generates a prompt for each image with Combo mode and Mix mode combined a maximum of 4 images in the list then moves to the next 4 and use it to run a job as many times as your batch count is set. the settings are taken from the yaml file"}),
                "batch_count": ("INT", {"default": 1, "tooltip": "Number of images to generate. only for create, edit and variations strategies."}),
                "external_api_key": ("STRING", {"default": "", "tooltip": "If this is not empty, it will be used instead of the API key from the .env file. Make sure it is empty to use the .env file."}),
                "Omni": ("OMNI", {"default": None, "tooltip": "Additional input for the selected tool."}),
            },
            "hidden": {
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "tooltip": "Controls randomness in output generation. Higher values increase creativity but may reduce coherence."}),
                "top_k": ("INT", {"default": 40, "tooltip": "Limits the next token selection to the K most likely tokens."}),
                "top_p": ("FLOAT", {"default": 0.9, "tooltip": "Cumulative probability cutoff for token selection."}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "tooltip": "Penalizes repetition in generated text."}),
                "precision": (["fp16", "fp32", "bf16"], {"tooltip": "Select preccision on Transformer models."}),
                "attention": (["sdpa", "flash_attention_2", "xformers"], {"tooltip": "Select attention mechanism on Transformer models."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "OMNI", "IMAGE", "MASK")
    RETURN_NAMES = ("question", "response", "negative", "omni", "generated_images", "mask")

    FUNCTION = "process_image_wrapper"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    async def process_image(
        self,
        llm_provider: str,
        llm_model: str,
        base_ip: str,
        port: str,
        user_prompt: str,
        strategy: str = "normal",
        images=None,
        messages=None,
        prime_directives: Optional[str] = None,
        profiles: Optional[str] = None,
        embellish_prompt: Optional[str] = None,
        style_prompt: Optional[str] = None,
        neg_prompt: Optional[str] = None,
        stop_string: Optional[str] = None,
        max_tokens: int = 2048,
        seed: int = 0,
        random: bool = False,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        keep_alive: bool = False,
        clear_history: bool = False,
        history_steps: int = 10,
        external_api_key: str = "",
        precision: str = "fp16",
        attention: str = "sdpa",
        Omni: Optional[str] = None,
        aspect_ratio: str = "1:1",
        mask: Optional[torch.Tensor] = None,
        batch_count: int = 4,
        auto: bool = False,
        auto_mode: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        try:
            # Initialize variables at the start
            formatted_response = None
            generated_images = None
            generated_masks = None
            tool_output = None

            if external_api_key != "":
                llm_api_key = external_api_key
            else:
                llm_api_key = get_api_key(f"{llm_provider.upper()}_API_KEY", llm_provider)
            print(f"LLM API key: {llm_api_key[:5]}...")

            # Validate LLM model
            validate_models(llm_model, llm_provider, "LLM", base_ip, port, llm_api_key)

            # Handle history
            if clear_history:
                messages = []
            elif history_steps > 0:
                messages = messages[-history_steps:]


            # Handle stop
            if stop_string is None or stop_string == "None":
                stop_content = None
            else:
                stop_content = self.stop_strings.get(stop_string, None)
            stop = stop_content

            if llm_provider not in ["ollama", "llamacpp", "vllm", "lmstudio", "gemeni"]:
                if llm_provider == "kobold":
                    stop = stop_content + \
                        ["\n\n\n\n\n"] if stop_content else ["\n\n\n\n\n"]
                elif llm_provider == "mistral":
                    stop = stop_content + \
                        ["\n\n"] if stop_content else ["\n\n"]
                else:
                    stop = stop_content if stop_content else None

            # Prepare embellishments and styles
            embellish_content = self.embellish_prompts.get(embellish_prompt, "").strip() if embellish_prompt else ""
            style_content = self.style_prompts.get(style_prompt, "").strip() if style_prompt else ""
            neg_content = self.neg_prompts.get(neg_prompt, "").strip() if neg_prompt else ""
            profile_content = self.profiles.get(profiles, "")

            # Prepare system prompt
            if prime_directives is not None:
                system_message = prime_directives
            else:
                system_message= json.dumps(profile_content)

            omni = Omni
            strategy_name = strategy

            kwargs = {
                'batch_count': batch_count,
                'llm_provider': llm_provider,
                'base_ip': base_ip,
                'port': port,
                'llm_model': llm_model,
                'system_message': system_message,
                'seed': seed,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'random': random,
                'top_k': top_k,
                'top_p': top_p,
                'repeat_penalty': repeat_penalty,
                'stop': stop,
                'keep_alive': keep_alive,
                'llm_api_key': llm_api_key,
                'precision': precision,
                'attention': attention,
                'aspect_ratio': aspect_ratio,
                'neg_prompt': neg_prompt,
                'neg_content': neg_content,
                'formatted_response': formatted_response,
                'generated_images': generated_images,
                'generated_masks': generated_masks,
                'tool_output': tool_output,
            }

            # Prepare images and mask
            if images is not None:  
                current_images = images
            else:
                current_images = load_placeholder_image(self.placeholder_image_path)[0]
            if mask is not None:
                current_mask = mask
            else:
                current_mask = load_placeholder_image(self.placeholder_image_path)[1]

            if auto:
                try:
                    # Use the main auto mode processing function
                    result = await self.process_auto_mode(
                        images=current_images,
                        mask=current_mask,
                        messages=messages,
                        strategy=strategy,
                        auto_mode=auto_mode,
                        **kwargs
                    )
                    
                    if result:
                        return result
                    else:
                        return self.create_error_response(
                            "No results generated from auto mode processing.",
                            user_prompt
                        )

                except Exception as e:
                    logger.error(f"Error in auto mode processing: {str(e)}")
                    return self.create_error_response(str(e), user_prompt)

            else: 
                # Execute strategy-specific logic
                if strategy_name == "normal":
                    return await self.execute_normal_strategy(
                        user_prompt, current_images, current_mask, messages, embellish_content, style_content, **kwargs)
                elif strategy_name == "create":
                    return await self.execute_create_strategy(
                        user_prompt, current_mask, **kwargs)
                elif strategy_name == "omost":
                    return await self.execute_omost_strategy(
                        user_prompt, current_images, current_mask, omni, embellish_content, style_content, **kwargs)
                elif strategy_name == "variations":
                    return await self.execute_variations_strategy(
                        user_prompt, current_images, **kwargs)
                elif strategy_name == "edit":
                    return await self.execute_edit_strategy(
                        user_prompt, current_images, current_mask, **kwargs)
                else:
                    raise ValueError(f"Unsupported strategy: {strategy_name}")

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            return {
                "Question": kwargs.get("user_prompt", ""),
                "Response": f"Error: {str(e)}",
                "Negative": "",
                "Tool_Output": None,
                "Retrieved_Image": (
                    images[0]
                    if images is not None and len(images) > 0
                    else load_placeholder_image(self.placeholder_image_path)[0]
                ),
                "Mask": (
                    torch.ones((images[0].shape[0], 1))
                    if images is not None and len(images) > 0
                    else load_placeholder_image(self.placeholder_image_path)[1]
                ),
            }
   
    async def process_auto_mode(self, images, mask, messages, strategy, auto_mode=True, embellish_content="", style_content="", **kwargs):
        """
        Main auto mode processing function that handles both mix and combo modes.
        
        Args:
            images: Input images
            mask: Input mask
            strategy: Strategy to use
            auto_mode: If True, use mix mode; if False, use combo mode
            messages: Message history (optional)
            embellish_content: Optional embellishment text
            style_content: Optional style text
            **kwargs: Additional arguments including:
                - batch_count: Number of variations to generate
                - llm_provider, llm_model, etc.
            
        Returns:
            Combined results from all batches
        """
        try:
            # Determine batch size based on auto_mode
            batch_size = 4 if auto_mode else 1

            # Process images into batches
            image_batches, mask_batches = process_auto_mode_images(
                images=images,
                mask=mask,
                batch_size=batch_size
            )

            results = []
            batch_count = kwargs.get('batch_count', 4)  # Use provided batch_count or default to 4

            # Process each batch
            for img_batch, mask_batch in zip(image_batches, mask_batches):
                # Generate prompt for current batch
                combo_prompt = await self.generate_combo_prompts(
                    images=img_batch,
                    settings_dict=None
                )

                if combo_prompt:
                    # Generate specified number of variations
                    for _ in range(batch_count):
                        batch_result = await self.process_auto_batch(
                            batch_images=img_batch,
                            batch_mask=mask_batch,
                            strategy=strategy,
                            prompt=combo_prompt,
                            messages=messages,
                            embellish_content=embellish_content,
                            style_content=style_content,
                            **kwargs  # Pass batch_count and other kwargs
                        )
                        if batch_result:
                            results.append(batch_result)

            # Combine results
            if results:
                combined_response = {
                    "Question": "\n".join(r.get("Question", "") for r in results),
                    "Response": "\n".join(r.get("Response", "") for r in results),
                    "Negative": "\n".join(r.get("Negative", "") for r in results),
                    "Tool_Output": [r.get("Tool_Output") for r in results],
                    "Retrieved_Image": images,
                    "Mask": mask
                }
                return combined_response

            # Return error response if no valid results
            return self.create_error_response("No valid results generated from auto mode processing.")

        except Exception as e:
            logger.error(f"Error in auto mode processing: {str(e)}")
            return self.create_error_response(str(e))

    async def process_auto_batch(self, batch_images, batch_mask, strategy, prompt, messages, 
                            embellish_content="", style_content="", **kwargs):
        """
        Process a single batch in auto mode.
        
        Args:
            batch_images: Tensor of batch images [B,H,W,C] where B <= batch_size
            batch_mask: Tensor of batch masks [B,H,W,1]
            strategy: Strategy to use
            prompt: Generated prompt
            messages: Message history
            embellish_content: Optional embellishment text
            style_content: Optional style text
            **kwargs: Additional strategy parameters
            
        Returns:
            Dict containing strategy results
        """
        try:
            # Ensure mask has correct dimensions [B,H,W,1]
            if batch_mask is not None:
                if len(batch_mask.shape) != 4:  
                    batch_mask = batch_mask.reshape(batch_mask.shape[0], 
                                              batch_mask.shape[1],
                                              batch_mask.shape[2], 1)
                # Add safety check for number of channels
                if batch_mask.shape[-1] != 1:
                    batch_mask = batch_mask[..., :1]  

            # Ensure we don't pass batch_count twice
            batch_kwargs = kwargs.copy()
            batch_kwargs.pop('batch_count', None)
            
            if strategy == "normal":
                return await self.execute_normal_strategy(
                    user_prompt=prompt,
                    current_images=batch_images,
                    current_mask=batch_mask,
                    messages=messages,
                    embellish_content=embellish_content,
                    style_content=style_content,
                    batch_count=1,  # Process one at a time
                    **batch_kwargs
                )
            elif strategy == "omost":
                return await self.execute_omost_strategy(
                    user_prompt=prompt,
                    current_images=batch_images,
                    current_mask=batch_mask,
                    omni=kwargs.get('omni'),
                    embellish_content=embellish_content,
                    style_content=style_content,
                    batch_count=1,  # Process one at a time
                    **batch_kwargs
                )
            else:
                raise ValueError(f"Unsupported strategy for auto mode: {strategy}")
        
        except Exception as e:
            logger.error(f"Error processing auto batch: {str(e)}")
            return None

    async def execute_normal_strategy(self, user_prompt, current_images, current_mask, 
                            messages, embellish_content, style_content, **kwargs):
        try:
            formatted_responses = []
            final_prompts = []
            final_negative_prompts = []
            print(kwargs.get('batch_count', 1))
            
            # Process batch_count times
            for _ in range(kwargs.get('batch_count', 1)):
                response = await send_request(
                    llm_provider=kwargs.get('llm_provider'),
                    base_ip=kwargs.get('base_ip'),
                    port=kwargs.get('port'),
                    images=current_images,
                    llm_model=kwargs.get('llm_model'),
                    system_message=kwargs.get('system_message'),
                    user_message=user_prompt,
                    messages=messages,
                    seed=kwargs.get('seed'),
                    temperature=kwargs.get('temperature'),
                    max_tokens=kwargs.get('max_tokens'),
                    random=kwargs.get('random'),
                    top_k=kwargs.get('top_k'),
                    top_p=kwargs.get('top_p'),
                    repeat_penalty=kwargs.get('repeat_penalty'),
                    stop=kwargs.get('stop'),
                    keep_alive=kwargs.get('keep_alive'),
                    llm_api_key=kwargs.get('llm_api_key'),
                    precision=kwargs.get('precision'),
                    attention=kwargs.get('attention'),
                    aspect_ratio=kwargs.get('aspect_ratio'),
                    strategy="normal",
                    batch_count=1,
                    mask=current_mask
                )

                if not response:
                    continue

                # Process response
                cleaned_response = clean_text(response)
                final_prompt = f"{embellish_content} {cleaned_response} {style_content}".strip()
                final_prompts.append(final_prompt)

                # Handle negative prompts
                if kwargs.get('neg_prompt') == "AI_Fill":
                    neg_prompt = await self.generate_negative_prompt(cleaned_response, images=current_images, **kwargs)
                    final_negative_prompts.append(neg_prompt)
                else:
                    final_negative_prompts.append(kwargs.get('neg_content', ''))

            # Combine all responses
            formatted_response = "\n".join(final_prompts)
            formatted_negative = "\n".join(final_negative_prompts)

            if kwargs.get('keep_alive') and formatted_response:
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": formatted_response})

            return {
                "Question": user_prompt,
                "Response": formatted_response,
                "Negative": formatted_negative,
                "Tool_Output": None,
                "Retrieved_Image": current_images,
                "Mask": current_mask
            }

        except Exception as e:
            logger.error(f"Error in normal strategy: {str(e)}")
            return self.create_error_response(str(e), user_prompt)

    async def execute_omost_strategy(self, user_prompt, current_images, current_mask,
                             omni, embellish_content="", style_content="", **kwargs):
        """Execute OMOST strategy with batch processing and proper negative prompt generation"""
        try:
            batch_count = kwargs.get('batch_count', 1)
            messages = []
            system_prompt = self.profiles.get("IF_Omost")
            final_prompts = []
            final_negative_prompts = []
            results = []

            # Process batch_count times
            for batch_idx in range(batch_count):
                try:
                    # Get LLM response
                    llm_response = await send_request(
                        llm_provider=kwargs.get('llm_provider'),
                        base_ip=kwargs.get('base_ip'),
                        port=kwargs.get('port'),
                        images=current_images,
                        llm_model=kwargs.get('llm_model'),
                        system_message=system_prompt,
                        user_message=user_prompt,
                        messages=messages,
                        seed=kwargs.get('seed', 0) + batch_idx if kwargs.get('seed', 0) != 0 else kwargs.get('seed', 0),
                        temperature=kwargs.get('temperature', 0.7),
                        max_tokens=kwargs.get('max_tokens', 2048),
                        random=kwargs.get('random', False),
                        top_k=kwargs.get('top_k', 40),
                        top_p=kwargs.get('top_p', 0.9),
                        repeat_penalty=kwargs.get('repeat_penalty', 1.1),
                        stop=kwargs.get('stop', None),
                        keep_alive=kwargs.get('keep_alive', False),
                        llm_api_key=kwargs.get('llm_api_key'),
                        precision=kwargs.get('precision', 'fp16'),
                        attention=kwargs.get('attention', 'sdpa'),
                        aspect_ratio=kwargs.get('aspect_ratio', '1:1'),
                        strategy="omost",
                        batch_count=1,
                        mask=current_mask
                    )

                    if not llm_response:
                        logger.warning(f"No response from LLM in batch {batch_idx}")
                        continue

                    # Process LLM response with OMOST tool
                    cleaned_response = clean_text(llm_response)
                    if isinstance(cleaned_response, list):
                        cleaned_response = "\n".join(cleaned_response)
                    final_prompt = f"{embellish_content} {cleaned_response} {style_content}".strip()
                    final_prompts.append(final_prompt)

                    tool_result = await omost_function({
                        "name": "omost_tool", 
                        "description": "Analyzes images composition and generates a Canvas representation.",
                        "system_prompt": system_prompt,
                        "input": user_prompt,
                        "llm_response": llm_response,
                        "function_call": None,
                        "omni_input": omni
                    })

                    if kwargs.get('neg_prompt') == "AI_Fill":
                        neg_prompt = await self.generate_negative_prompt(cleaned_response, images=current_images, **kwargs)
                        final_negative_prompts.append(neg_prompt)
                    else:
                        final_negative_prompts.append(kwargs.get('neg_content', ''))

                    # Extract canvas conditioning if available
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            logger.warning(f"OMOST tool warning in batch {batch_idx}: {tool_result['error']}")
                            continue

                        canvas_cond = tool_result.get("canvas_conditioning")
                        if canvas_cond is not None:
                            # Store result for this batch
                            results.append({
                                "Question": user_prompt,
                                "Response": final_prompt,
                                "Negative": final_negative_prompts[-1],
                                "Tool_Output": canvas_cond,
                                "Retrieved_Image": current_images,
                                "Mask": current_mask
                            })

                except Exception as batch_error:
                    logger.error(f"Error in OMOST batch {batch_idx}: {str(batch_error)}")
                    continue

            # Handle results aggregation
            if not results:
                return self.create_error_response("No valid results generated", user_prompt)

            # Combine all results
            combined_response = {
                "Question": user_prompt,
                "Response": "\n".join(final_prompts),
                "Negative": "\n".join(final_negative_prompts),
                "Tool_Output": [r.get("Tool_Output") for r in results],
                "Retrieved_Image": current_images,
                "Mask": current_mask
            }

            return combined_response

        except Exception as e:
            logger.error(f"Error in OMOST strategy: {str(e)}")
            return self.create_error_response(str(e), user_prompt)

    async def execute_create_strategy(self, user_prompt, current_mask, **kwargs):
        try:
            # Create strategy - no input images needed
            messages = []
            api_response = await send_request(
                llm_provider=kwargs.get('llm_provider'),
                base_ip=kwargs.get('base_ip'),
                port=kwargs.get('port'),
                images=None,  # No input images needed for create
                llm_model=kwargs.get('llm_model'),
                system_message=kwargs.get('system_message'),
                user_message=user_prompt,
                messages=messages,
                seed=kwargs.get('seed', 0),
                temperature=kwargs.get('temperature'),
                max_tokens=kwargs.get('max_tokens'),
                random=kwargs.get('random'),
                top_k=kwargs.get('top_k'),
                top_p=kwargs.get('top_p'),
                repeat_penalty=kwargs.get('repeat_penalty'),
                stop=kwargs.get('stop'),
                keep_alive=kwargs.get('keep_alive'),
                llm_api_key=kwargs.get('llm_api_key'),
                precision=kwargs.get('precision'),
                attention=kwargs.get('attention'),
                aspect_ratio=kwargs.get('aspect_ratio'),
                strategy="create",
                batch_count= 1,
                mask=current_mask
            )

            # Extract base64 images from response
            all_base64_images = []
            if isinstance(api_response, dict) and "images" in api_response:
                base64_images = api_response.get("images", [])
                all_base64_images.extend(base64_images if isinstance(base64_images, list) else [base64_images])

            # Process the images if we have any
            if all_base64_images:
                # Prepare data for processing
                image_data = {
                    "data": [{"b64_json": img} for img in all_base64_images]
                }

                # Process images
                images_tensor, mask_tensor = process_images_for_comfy(
                    image_data,
                    placeholder_image_path=self.placeholder_image_path,
                    response_key="data",
                    field_name="b64_json"
                )

                logger.debug(f"Retrieved_Image tensor shape: {images_tensor.shape}")

                return {
                    "Question": user_prompt,
                    "Response": f"Create image{'s' if len(all_base64_images) > 1 else ''} successfully generated.",
                    "Negative": kwargs.get('neg_content', ''),
                    "Tool_Output": all_base64_images,
                    "Retrieved_Image": images_tensor,
                    "Mask": mask_tensor
                }
            else:
                # No images were generated
                image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
                return {
                    "Question": user_prompt,
                    "Response": "No images were generated in create strategy",
                    "Negative": kwargs.get('neg_content', ''),
                    "Tool_Output": None,
                    "Retrieved_Image": image_tensor,
                    "Mask": mask_tensor
                }

        except Exception as e:
            logger.error(f"Error in create strategy: {str(e)}")
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return {
                "Question": user_prompt,
                "Response": f"Error in create strategy: {str(e)}",
                "Negative": kwargs.get('neg_content', ''),
                "Tool_Output": None,
                "Retrieved_Image": image_tensor,
                "Mask": mask_tensor
            }

    async def execute_variations_strategy(self, user_prompt, images, **kwargs):
        """Core implementation of variations strategy"""
        try:
            batch_count = kwargs.get('batch_count', 1)
            messages = []
            api_responses = []

            # Prepare input images
            input_images = prepare_batch_images(images)

            # Process each input image
            for img in input_images:
                try:
                    # Send request for variations
                    api_response = await send_request(
                        images=img,
                        user_message=user_prompt,
                        messages=messages,
                        strategy="variations",
                        batch_count=batch_count,
                        mask=None,  # Variations don't use masks
                        **kwargs
                    )
                    if api_response:
                        api_responses.append(api_response)
                except Exception as e:
                    logger.error(f"Error processing image variation: {str(e)}")
                    continue

            # Extract and process base64 images from responses
            all_base64_images = []
            for response in api_responses:
                if isinstance(response, dict) and "images" in response:
                    base64_images = response.get("images", [])
                    if isinstance(base64_images, list):
                        all_base64_images.extend(base64_images)
                    else:
                        all_base64_images.append(base64_images)

            # Process the generated images
            if all_base64_images:
                # Prepare data for processing
                image_data = {
                    "data": [{"b64_json": img} for img in all_base64_images]
                }

                # Convert to tensors
                images_tensor, mask_tensor = process_images_for_comfy(
                    image_data,
                    placeholder_image_path=self.placeholder_image_path,
                    response_key="data",
                    field_name="b64_json"
                )

                logger.debug(f"Variations image tensor shape: {images_tensor.shape}")

                return self.create_strategy_response(
                    user_prompt=user_prompt,
                    response_text=f"Generated {len(all_base64_images)} variations successfully.",
                    images_tensor=images_tensor,
                    mask_tensor=mask_tensor,
                    neg_content=kwargs.get('neg_content', ''),
                    tool_output=all_base64_images
                )
            else:
                # No variations were generated
                image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
                return self.create_strategy_response(
                    user_prompt=user_prompt,
                    response_text="No variations were generated",
                    images_tensor=image_tensor,
                    mask_tensor=mask_tensor,
                    neg_content=kwargs.get('neg_content', '')
                )

        except Exception as e:
            logger.error(f"Error in variations strategy: {str(e)}")
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return self.create_strategy_response(
                user_prompt=user_prompt,
                response_text=f"Error in variations strategy: {str(e)}",
                images_tensor=image_tensor,
                mask_tensor=mask_tensor,
                neg_content=kwargs.get('neg_content', '')
            )

    async def execute_edit_strategy(self, user_prompt, images, mask, **kwargs):
        """Core implementation of edit strategy"""
        try:
            batch_count = kwargs.get('batch_count', 1)
            messages = []
            api_responses = []

            # Prepare input images and masks
            input_images = prepare_batch_images(images)
            input_masks = prepare_batch_images(mask) if mask is not None else [None] * len(input_images)

            # Process each image-mask pair
            for img, msk in zip(input_images, input_masks):
                try:
                    # Send request for edit
                    api_response = await send_request(
                        images=img,
                        user_message=user_prompt,
                        messages=messages,
                        strategy="edit",
                        batch_count=batch_count,
                        mask=msk,
                        **kwargs
                    )
                    if api_response:
                        api_responses.append(api_response)
                except Exception as e:
                    logger.error(f"Error processing image-mask pair: {str(e)}")
                    continue

            # Extract and process base64 images from responses
            all_base64_images = []
            for response in api_responses:
                if isinstance(response, dict) and "images" in response:
                    base64_images = response.get("images", [])
                    if isinstance(base64_images, list):
                        all_base64_images.extend(base64_images)
                    else:
                        all_base64_images.append(base64_images)

            # Process the edited images
            if all_base64_images:
                # Prepare data for processing
                image_data = {
                    "data": [{"b64_json": img} for img in all_base64_images]
                }

                # Convert to tensors
                images_tensor, mask_tensor = process_images_for_comfy(
                    image_data,
                    placeholder_image_path=self.placeholder_image_path,
                    response_key="data",
                    field_name="b64_json"
                )

                logger.debug(f"Edited image tensor shape: {images_tensor.shape}")

                return self.create_strategy_response(
                    user_prompt=user_prompt,
                    response_text=f"Successfully edited {len(all_base64_images)} images.",
                    images_tensor=images_tensor,
                    mask_tensor=mask_tensor,
                    neg_content=kwargs.get('neg_content', ''),
                    tool_output=all_base64_images
                )
            else:
                # No edits were generated
                image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
                return self.create_strategy_response(
                    user_prompt=user_prompt,
                    response_text="No edited images were generated",
                    images_tensor=image_tensor,
                    mask_tensor=mask_tensor,
                    neg_content=kwargs.get('neg_content', '')
                )

        except Exception as e:
            logger.error(f"Error in edit strategy: {str(e)}")
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return self.create_strategy_response(
                user_prompt=user_prompt,
                response_text=f"Error in edit strategy: {str(e)}",
                images_tensor=image_tensor,
                mask_tensor=mask_tensor,
                neg_content=kwargs.get('neg_content', '')
            )

    def get_models(self, engine, base_ip, port, api_key=None):
        return get_models(engine, base_ip, port, api_key)

    def load_presets(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading presets from {file_path}: {e}")
            return {}

    def validate_outputs(self, outputs):
        """Helper to validate output types match expectations"""
        if len(outputs) != len(self.RETURN_TYPES):
            raise ValueError(
                f"Expected {len(self.RETURN_TYPES)} outputs, got {len(outputs)}"
            )

        for i, (output, expected_type) in enumerate(zip(outputs, self.RETURN_TYPES)):
            if output is None and expected_type in ["IMAGE", "MASK"]:
                raise ValueError(
                    f"Output {i} ({self.RETURN_NAMES[i]}) cannot be None for type {expected_type}"
                )

    async def generate_combo_prompts(self, images, settings_dict=None, **kwargs):
        """Generate combo prompts using saved or provided settings."""
        try:
            # If no settings provided, load from file
            if settings_dict is None:
                settings_dict = load_combo_settings(self.combo_presets_dir)

            if not settings_dict:
                raise ValueError("No combo settings available")

            # Get the profile content
            profile_name = settings_dict.get('profile', 'IF_PromptMKR')
            profile_content = self.profiles.get(profile_name, {}).get('instruction', '')

            # If 'prime_directives' is empty, use the profile content
            if not settings_dict.get('prime_directives'):
                settings_dict['prime_directives'] = profile_content

            # Extract API key
            llm_provider = settings_dict.get('llm_provider', '')
            if settings_dict.get('external_api_key'):
                llm_api_key = settings_dict['external_api_key']
            else:
                llm_api_key = get_api_key(f"{llm_provider.upper()}_API_KEY", llm_provider)

            # Send request using settings
            response = await send_request(
                llm_provider=llm_provider,
                base_ip=settings_dict.get('base_ip', 'localhost'),
                port=settings_dict.get('port', '11434'),
                images=images,
                llm_model=settings_dict.get('llm_model', ''),
                system_message=settings_dict.get('prime_directives', ''),
                user_message=settings_dict.get('user_prompt', ''),
                messages=[],  # Empty list for fresh context
                seed=settings_dict.get('seed', 0),
                temperature=settings_dict.get('temperature', 0.7),
                max_tokens=settings_dict.get('max_tokens', 2048),
                random=settings_dict.get('random', False),
                top_k=settings_dict.get('top_k', 40),
                top_p=settings_dict.get('top_p', 0.9),
                repeat_penalty=settings_dict.get('repeat_penalty', 1.1),
                stop=settings_dict.get('stop_string'),
                keep_alive=settings_dict.get('keep_alive', False),
                llm_api_key=llm_api_key,
                precision=settings_dict.get('precision', 'fp16'),
                attention=settings_dict.get('attention', 'sdpa'),
                aspect_ratio=settings_dict.get('aspect_ratio', '1:1'),
                strategy="normal",
                mask=None,
                batch_count=1
            )

            if isinstance(response, dict):
                return response.get('response', '')
            return response

        except Exception as e:
            logger.error(f"Error generating combo prompts: {str(e)}")
            return ""

    def process_image_wrapper(self, **kwargs):
        """Wrapper to handle async execution of process_image"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Ensure images is present in kwargs
            if 'images' not in kwargs:
                raise ValueError("Input images are required")

            # Ensure all other required parameters are present
            required_params = ['llm_provider', 'llm_model', 'base_ip', 'port', 'user_prompt']
            missing_params = [p for p in required_params if p not in kwargs]
            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

            # Get the result from process_image
            result = loop.run_until_complete(self.process_image(**kwargs))

            # Extract values in the correct order matching RETURN_TYPES
            prompt = result.get("Response", "")  # This is the formatted prompt
            response = result.get("Question", "")  # Original question/prompt
            negative = result.get("Negative", "")
            omni = result.get("Tool_Output")
            retrieved_image = result.get("Retrieved_Image")
            mask = result.get("Mask")

            # Ensure we have valid image and mask tensors
            if retrieved_image is None or not isinstance(retrieved_image, torch.Tensor):
                retrieved_image, mask = load_placeholder_image(self.placeholder_image_path)

            # Ensure mask has correct format
            if mask is None:
                mask = torch.ones((retrieved_image.shape[0], 1, retrieved_image.shape[2], retrieved_image.shape[3]), 
                                dtype=torch.float32,
                                device=retrieved_image.device)

            # Return tuple matching RETURN_TYPES order: ("STRING", "STRING", "STRING", "OMNI", "IMAGE", "MASK")
            return (
                response,  # First STRING (question/prompt)
                prompt,    # Second STRING (generated response)
                negative,  # Third STRING (negative prompt)
                omni,      # OMNI
                retrieved_image,  # IMAGE
                mask       # MASK
            )

        except Exception as e:
            logger.error(f"Error in process_image_wrapper: {str(e)}")
            # Create fallback values
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return (
                kwargs.get("user_prompt", ""),  # Original prompt
                f"Error: {str(e)}",            # Error message as response
                "",                            # Empty negative prompt
                None,                          # No OMNI data
                image_tensor,                  # Placeholder image
                mask_tensor                    # Default mask
            )

    def create_error_response(self, images, error_message, prompt=""):
        """Create standardized error response"""
        try:
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return {
                "Question": prompt,
                "Response": f"Error: {error_message}",
                "Negative": "",
                "Tool_Output": None,
                "Retrieved_Image": image_tensor,
                "Mask": mask_tensor
            }
        except Exception as e:
            logger.error(f"Error creating error response: {str(e)}")
            # Fallback error response without images
            return {
                "Question": prompt,
                "Response": f"Critical Error: {error_message}",
                "Negative": "",
                "Tool_Output": None,
                "Retrieved_Image": None,
                "Mask": None
            }

    async def generate_negative_prompt(
        self,
        prompt: str,
        images: List[Image.Image],
        **kwargs
    ) -> List[str]:
        """
        Generate negative prompts for the given input prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Generation parameters like seed, temperature etc
            
        Returns:
            List of generated negative prompts
        """
        try:
            if not prompt:
                return []
             
            # Get system message for negative prompts and ensure it's a string
            neg_system_message = self.profiles.get("IF_NegativePromptEngineer_V2", "")
            if isinstance(neg_system_message, dict):
                neg_system_message = json.dumps(neg_system_message)
            # Generate negative prompt using cleaned response
            neg_prompt = await send_request(
                llm_provider=kwargs.get('llm_provider'),
                base_ip=kwargs.get('base_ip'),
                port=kwargs.get('port'),
                images=images,  
                llm_model=kwargs.get('llm_model'),
                system_message=neg_system_message,
                user_message=f"Generate negative prompts for:\n{prompt}",
                messages=[],  # Fresh context for negative generation
                seed=kwargs.get('seed', 0),
                temperature=kwargs.get('temperature'),
                max_tokens=kwargs.get('max_tokens'),
                random=kwargs.get('random'),
                top_k=kwargs.get('top_k'),
                top_p=kwargs.get('top_p'),
                repeat_penalty=kwargs.get('repeat_penalty'),
                stop=kwargs.get('stop'),
                keep_alive=kwargs.get('keep_alive'),
                llm_api_key=kwargs.get('llm_api_key'),
            )
            if neg_prompt:
                return clean_text(neg_prompt)
            else:
                return kwargs.get('neg_content', '')
            
        except Exception as e:
            logger.error(f"Error generating negative prompts: {str(e)}")
            return ["Error generating negative prompt"] 

NODE_CLASS_MAPPINGS = {
    "IF_PROMPTImaGEN": IFPROMPTImaGEN
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_PROMPTImaGEN": "IF Prompt ImaGEN 🖼️"
}
