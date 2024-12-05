# IFLLMNode.py
import os
import sys
import json
import torch
import asyncio
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
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
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

try:
    import folder_paths
except ImportError:
    print("Error: Could not import folder_paths. Make sure ComfyUI core is in your Python path.")
    folder_paths = None

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/IF_LLM/get_llm_models")
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

            node = IFLLM()
            models = node.get_models(engine, base_ip, port, api_key)
            return web.json_response(models)
        
        except Exception as e:
            print(f"Error in get_llm_models_endpoint: {str(e)}")
            return web.json_response([], status=500)

    @PromptServer.instance.routes.post("/IF_LLM/add_routes")
    async def add_routes_endpoint(request):
        return web.json_response({"status": "success"})

    @PromptServer.instance.routes.post("/IF_LLM/save_combo_settings")
    async def save_combo_settings_endpoint(request):
        try:
            data = await request.json()
            
            # Convert UI settings to proper format
            settings = create_settings_from_ui(data)
            
            # Get node instance
            node = IFLLM()
            
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
    print("PromptServer.instance not available. Skipping route decoration for IF_LLM.")

class IFLLM:
    def __init__(self):
        self.strategies = "normal"
        # Initialize paths and load presets
        # Get the directory where the current script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build paths relative to the script location
        self.presets_dir = os.path.join(current_dir, "IF_AI", "presets")
        self.combo_presets_dir = os.path.join(self.presets_dir, "AutoCombo")
        # Load preset configurations
        self.profiles = self.load_presets(os.path.join(self.presets_dir, "profiles.json"))
        self.neg_prompts = self.load_presets(os.path.join(self.presets_dir, "neg_prompts.json"))
        self.embellish_prompts = self.load_presets(os.path.join(self.presets_dir, "embellishments.json"))
        self.style_prompts = self.load_presets(os.path.join(self.presets_dir, "style_prompts.json"))
        self.stop_strings = self.load_presets(os.path.join(self.presets_dir, "stop_strings.json"))

        # Initialize placeholder image path
        self.placeholder_image_path = os.path.join(self.presets_dir, "placeholder.png")

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
                "auto": ("BOOLEAN", {"default": False, "label_on": "Auto Is Enabled", "label_off": "Auto is Disabled", "tooltip": "If true, it generates auto promts based on the listed images click the save Auto settings to set the auto prompt generation file"}),
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

            tool_type = Omni
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
                'omni': tool_type,
            }

            # Prepare images and mask
            if images is not None:  
                current_images = images
            else:
                raise ValueError("No images provided you need to provide at least one image")
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
                        #self, images, masks, error_message, prompt=""
                        return self.create_error_response(
                            current_images,
                            current_mask,
                            "No results generated from auto mode processing.",
                            user_prompt
                        )

                except Exception as e:
                    logger.error(f"Error in auto mode processing: {str(e)}")
                    return self.create_error_response(
                            current_images,
                            current_mask,
                            "No results generated from auto mode processing.",
                            user_prompt
                        )

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
                        user_prompt, current_images, current_mask, embellish_content, style_content, **kwargs)
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
            return self.create_error_response(
                            current_images,
                            current_mask,
                            "No results generated from auto mode processing.",
                            user_prompt
                        )
   
    async def process_auto_mode(self, images, mask, messages, strategy, auto_mode=True, embellish_content="", style_content="", **kwargs):
        """
        Main auto mode processing function that preserves batch handling.
        """
        try:
            # Determine batch size based on mode
            batch_size = 4 if auto_mode else 1  

            # Process images into appropriate batches
            image_batches, mask_batches = process_auto_mode_images(
                images=images,
                mask=mask,
                batch_size=batch_size
            )

            all_results = []
            user_prompt = kwargs.get('user_prompt', '')
            batch_count = kwargs.get('batch_count', 1)
            
            # Process each image/mask batch
            for img_batch, mask_batch in zip(image_batches, mask_batches):

                for i in range(img_batch.size(0)):
                    single_img = img_batch[i:i+1]
                    single_mask = mask_batch[i:i+1]
                        
                    # Generate combo prompt once for this image
                    combo_prompt = await self.generate_combo_prompts(
                        images=single_img,
                        settings_dict=None
                    )
                        
                    # Process batch_count iterations for this image
                    for iteration in range(batch_count):
                        batch_results = await self.process_auto_batch(
                            batch_images=single_img,
                            batch_mask=single_mask,
                            strategy=strategy,
                            prompt=combo_prompt,
                            messages=messages,
                            embellish_content=embellish_content,
                            style_content=style_content,
                            **{**kwargs, 
                            'batch_count': 1,  # Process single iteration here
                            'seed': kwargs.get('seed', 0) + iteration if kwargs.get('seed') is not None else None
                            }
                        )
                            
                        if batch_results:
                            if isinstance(batch_results, list):
                                all_results.extend(batch_results)
                            else:
                                all_results.append(batch_results)

            if not all_results:
                return [{
                    "Question": user_prompt,
                    "Response": "No results generated",
                    "Negative": "",
                    "Tool_Output": None,
                    "Retrieved_Image": images,
                    "Mask": mask
                }]
                
            return all_results

        except Exception as e:
            logger.error(f"Error in process_auto_mode: {str(e)}")
            return [{
                "Question": kwargs.get('user_prompt', ''),
                "Response": f"Error: {str(e)}",
                "Negative": "",
                "Tool_Output": None,
                "Retrieved_Image": images,
                "Mask": mask
            }]

        
    async def process_auto_batch(self, batch_images, batch_mask, strategy, prompt, messages, 
                            embellish_content="", style_content="", **kwargs):
        """
        Process single iteration of auto mode batch.
        Batch count iterations are handled by process_auto_mode.
        """
        try:
            # Create clean kwargs without user_prompt
            batch_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in ['user_prompt']
            }
            
            # Execute strategy (should process just one iteration)
            if strategy == "normal":
                results = await self.execute_normal_strategy(
                    user_prompt=prompt,
                    current_images=batch_images,
                    current_mask=batch_mask,
                    messages=messages,
                    embellish_content=embellish_content,
                    style_content=style_content,
                    **batch_kwargs
                )
            elif strategy == "omost":
                results = await self.execute_omost_strategy(
                    user_prompt=prompt,
                    current_images=batch_images,
                    current_mask=batch_mask,
                    omni=kwargs.get('omni'),
                    embellish_content=embellish_content,
                    style_content=style_content,
                    **batch_kwargs
                )
            else:
                raise ValueError(f"Unsupported strategy for auto mode: {strategy}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error processing auto batch: {str(e)}")
            return None

    async def execute_normal_strategy(self, user_prompt, current_images, current_mask, 
                                messages, embellish_content, style_content, **kwargs):
        """
        Execute normal strategy with batch count handling.
        This can be called directly or through auto mode.
        """
        try:
            results = []
            # Keep batch_count for direct calls
            batch_count = kwargs.get('batch_count', 1)
            
            # Process batch_count times
            for i in range(batch_count):
                # Update seed for each iteration if using random seeding
                if kwargs.get('random', False) and 'seed' in kwargs:
                    base_seed = kwargs['seed']
                    if base_seed is not None:
                        current_seed = base_seed + i
                    else:
                        current_seed = kwargs['seed']
                else:
                    current_seed = kwargs.get('seed')

                response = await send_request(
                    llm_provider=kwargs.get('llm_provider'),
                    base_ip=kwargs.get('base_ip'),
                    port=kwargs.get('port'),
                    images=current_images,
                    llm_model=kwargs.get('llm_model'),
                    system_message=kwargs.get('system_message'),
                    user_message=user_prompt,
                    messages=messages,
                    seed=current_seed,
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
                    mask=current_mask
                )

                if not response:
                    continue

                cleaned_response = clean_text(response)
                final_prompt = "\n".join(filter(None, [
                    embellish_content.strip(),
                    cleaned_response.strip(),
                    style_content.strip()
                ]))
                
                if kwargs.get('neg_prompt') == "AI_Fill":
                    neg_prompt = await self.generate_negative_prompt(
                        cleaned_response, 
                        images=current_images, 
                        **kwargs
                    )
                else:
                    neg_prompt = kwargs.get('neg_content', '')

                results.append({
                    "Question": user_prompt,
                    "Response": final_prompt,
                    "Negative": neg_prompt,
                    "Tool_Output": None,
                    "Retrieved_Image": current_images,
                    "Mask": current_mask
                })

            # Keep message history if enabled
            if kwargs.get('keep_alive') and results:
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": results[-1]["Response"]})

            return results

        except Exception as e:
            logger.error(f"Error in normal strategy: {str(e)}")
            return [self.create_error_response(
                current_images,
                current_mask,
                "No results generated from normal strategy.",
                user_prompt
            )]
        
    async def execute_omost_strategy(self, user_prompt, current_images, current_mask,
                             omni, embellish_content="", style_content="", **kwargs):
        """Execute OMOST strategy with batch processing and proper negative prompt generation"""
        try:
            batch_count = kwargs.get('batch_count', 1)
            messages = []
            system_prompt = self.profiles.get("IF_Omost")
            results = []
            
            logger.debug(f"Processing {batch_count} batches in OMOST strategy")

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
                        mask=current_mask
                    )

                    if not llm_response:
                        logger.warning(f"No response from LLM in batch {batch_idx}")
                        continue

                    # Process LLM response
                    cleaned_response = clean_text(llm_response)
                    if isinstance(cleaned_response, list):
                        cleaned_response = "\n".join(cleaned_response)
                    final_prompt = "\n".join(filter(None, [
                        embellish_content.strip(),
                        cleaned_response.strip(),
                        style_content.strip()
                    ]))

                    tool_result = await omost_function({
                        "name": "omost_tool", 
                        "description": "Analyzes images composition and generates a Canvas representation.",
                        "system_prompt": system_prompt,
                        "input": user_prompt,
                        "llm_response": llm_response,
                        "function_call": None,
                        "omni_input": omni
                    })

                    # Handle negative prompt
                    if kwargs.get('neg_prompt') == "AI_Fill":
                        neg_prompt = await self.generate_negative_prompt(cleaned_response, images=current_images, **kwargs)
                    else:
                        neg_prompt = kwargs.get('neg_content', '')

                    # Extract canvas conditioning and create individual result
                    if isinstance(tool_result, dict):
                        if "error" in tool_result:
                            logger.warning(f"OMOST tool warning in batch {batch_idx}: {tool_result['error']}")
                            continue

                        canvas_cond = tool_result.get("canvas_conditioning")
                        if canvas_cond is not None:
                            results.append({
                                "Question": user_prompt,
                                "Response": final_prompt,
                                "Negative": neg_prompt,
                                "Tool_Output": canvas_cond,
                                "Retrieved_Image": current_images,
                                "Mask": current_mask
                            })

                except Exception as batch_error:
                    logger.error(f"Error in OMOST batch {batch_idx}: {str(batch_error)}")
                    continue

            # Keep message history if enabled
            if kwargs.get('keep_alive') and results:
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": results[-1]["Response"]})

            logger.debug(f"Generated {len(results)} results in OMOST strategy")

            # Handle results
            if not results:
                return [self.create_error_response(
                            current_images,
                            current_mask,
                            "No valid results generated",
                            user_prompt
                        )]

            # Return list of individual results
            return results

        except Exception as e:
            logger.error(f"Error in OMOST strategy: {str(e)}")
            return [self.create_error_response(
                            current_images,
                            current_mask,
                            "No valid results generated",
                            user_prompt
                        )]

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
                return self.create_error_response(
                            image_tensor,
                            mask_tensor,
                            "No images were generated in create strategy",
                            user_prompt
                        )

        except Exception as e:
            logger.error(f"Error in create strategy: {str(e)}")
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return self.create_error_response(
                            image_tensor,
                            mask_tensor,
                            f"Error in create strategy: {str(e)}",
                            user_prompt
                        )

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

                return {
                    "Question": user_prompt,
                    "Response": f"Generated {len(all_base64_images)} variations successfully.",
                    "Negative": kwargs.get('neg_content', ''),
                    "Tool_Output": all_base64_images,
                    "Retrieved_Image": images_tensor,
                    "Mask": mask_tensor
                }
            else:
                # No variations were generated
                image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
                return self.create_error_response(
                            image_tensor,
                            mask_tensor,
                            "No variations were generated",
                            user_prompt
                        )

        except Exception as e:
            logger.error(f"Error in variations strategy: {str(e)}")
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return self.create_error_response(
                            image_tensor,
                            mask_tensor,
                            f"Error in variations strategy: {str(e)}",
                            user_prompt
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

                return {
                    "Question": user_prompt,
                    "Response": f"Generated {len(all_base64_images)} variations successfully.",
                    "Negative": kwargs.get('neg_content', ''),
                    "Tool_Output": all_base64_images,
                    "Retrieved_Image": images_tensor,
                    "Mask": mask_tensor
                }
            else:
                # No edits were generated
                image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
                return self.create_error_response(
                            image_tensor,
                            mask_tensor,
                            "No edited images were generated",
                            user_prompt
                        )

        except Exception as e:
            logger.error(f"Error in edit strategy: {str(e)}")
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return self.create_error_response(
                            image_tensor,
                            mask_tensor,
                            f"Error in edit strategy: {str(e)}",
                            user_prompt
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
        try:
            if settings_dict is None:
                settings_dict = load_combo_settings(self.combo_presets_dir)

            if not settings_dict:
                raise ValueError("No combo settings available")

            # Get the profile content
            profile_name = settings_dict.get('profile', 'IF_PromptMKR')
            profile_content = self.profiles.get(profile_name, {}).get('instruction', '')

            if not settings_dict.get('prime_directives'):
                settings_dict['prime_directives'] = profile_content

            # Extract API key
            llm_provider = settings_dict.get('llm_provider', '')
            if settings_dict.get('external_api_key'):
                llm_api_key = settings_dict['external_api_key']
            else:
                llm_api_key = get_api_key(f"{llm_provider.upper()}_API_KEY", llm_provider)

            # Create request parameters with correct mappings
            request_params = {
                'llm_provider': settings_dict.get('llm_provider', ''),
                'base_ip': settings_dict.get('base_ip', 'localhost'),
                'port': settings_dict.get('port', '11434'),
                'images': images,
                'llm_model': settings_dict.get('llm_model', ''),
                'system_message': settings_dict.get('prime_directives', ''),  # Map prime_directives to system_message
                'user_message': settings_dict.get('user_prompt', ''),  # Map user_prompt to user_message
                'messages': [],
                'seed': settings_dict.get('seed', None),
                'temperature': settings_dict.get('temperature', 0.7),
                'max_tokens': settings_dict.get('max_tokens', 2048),
                'random': settings_dict.get('random', False),
                'top_k': settings_dict.get('top_k', 40),
                'top_p': settings_dict.get('top_p', 0.9),
                'repeat_penalty': settings_dict.get('repeat_penalty', 1.1),
                'stop': settings_dict.get('stop_string', None),  # Map stop_string to stop
                'keep_alive': settings_dict.get('keep_alive', False),
                'llm_api_key': llm_api_key,
                'precision': settings_dict.get('precision', 'fp16'),
                'attention': settings_dict.get('attention', 'sdpa'),
                'aspect_ratio': settings_dict.get('aspect_ratio', '1:1'),
                'strategy': 'normal',
                'mask': None,
                'batch_count': settings_dict.get('batch_count', 1)
            }

            response = await send_request(**request_params)

            if isinstance(response, dict):
                return response.get('response', '')
            return response

        except Exception as e:
            logger.error(f"Error generating combo prompts: {str(e)}")
            return ""

    def process_image_wrapper(self, **kwargs):
        """Wrapper to handle async execution of process_image"""
        try:
            # Attempt to get the current event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Create a new event loop if one doesn't exist
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Validate required inputs
            if 'images' not in kwargs:
                raise ValueError("Input images are required")

            required_params = ['llm_provider', 'llm_model', 'base_ip', 'port', 'user_prompt']
            missing_params = [p for p in required_params if p not in kwargs]
            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

            # Execute the asynchronous process_image method
            result = loop.run_until_complete(self.process_image(**kwargs))

            # Initialize aggregation lists
            responses = []
            prompts = []
            negatives = []
            omnis = []
            retrieved_images = []
            masks = []

            # Aggregate results based on their type
            if isinstance(result, list):
                if not result:
                    raise ValueError("No results generated")

                for result_item in result:
                    if isinstance(result_item, dict):
                        prompts.append(result_item.get("Response", ""))
                        responses.append(result_item.get("Question", ""))
                        negatives.append(result_item.get("Negative", ""))
                        omnis.append(result_item.get("Tool_Output"))
                        retrieved_images.append(result_item.get("Retrieved_Image"))
                        masks.append(result_item.get("Mask"))
                    else:
                        raise ValueError(f"Unexpected result format: {type(result_item)}")

            elif isinstance(result, dict):
                prompts.append(result.get("Response", ""))
                responses.append(result.get("Question", ""))
                negatives.append(result.get("Negative", ""))
                omnis.append(result.get("Tool_Output"))
                retrieved_images.append(result.get("Retrieved_Image"))
                masks.append(result.get("Mask"))
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")

            # Concatenate image tensors if present
            if retrieved_images:
                retrieved_images_tensor = torch.cat(retrieved_images, dim=0)  # Shape: [batch, 3, H, W]
            else:
                retrieved_images_tensor, _ = load_placeholder_image(self.placeholder_image_path)

            # Concatenate mask tensors if present
            if masks:
                masks_tensor = torch.cat(masks, dim=0)  # Shape: [batch, 1, H, W]
            else:
                _, masks_tensor = load_placeholder_image(self.placeholder_image_path)

            # Debug logging for verification
            for idx in range(len(retrieved_images)):
                logger.debug(f"Result {idx + 1}: Retrieved image type: {type(retrieved_images[idx])}")
                if isinstance(retrieved_images[idx], torch.Tensor):
                    logger.debug(f"Result {idx + 1}: Retrieved image shape: {retrieved_images[idx].shape}")
                logger.debug(f"Result {idx + 1}: Mask type: {type(masks[idx])}")
                if isinstance(masks[idx], torch.Tensor):
                    logger.debug(f"Result {idx + 1}: Mask shape: {masks[idx].shape}")

            # Ensure masks_tensor has the expected shape
            # Expected: [batch_size, 1, H, W]
            # If masks_tensor is not in the correct shape, adjust accordingly
            if masks_tensor.dim() == 3:
                masks_tensor = masks_tensor.unsqueeze(1)  # Add channel dimension if missing

            # Return the aggregated results
            return (
                responses,             # List of STRING (questions/prompts)
                prompts,               # List of STRING (generated responses)
                negatives,             # List of STRING (negative prompts)
                omnis,                 # List of OMNI
                retrieved_images_tensor,  # Concatenated IMAGE tensors [batch, 3, H, W]
                masks_tensor               # Concatenated MASK tensors [batch, 1, H, W]
            )

        except Exception as e:
            logger.error(f"Error in process_image_wrapper: {str(e)}")
            # Create fallback values as lists to match RETURN_TYPES
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return (
                [kwargs.get("user_prompt", "")],               # List containing original prompt
                [f"Error: {str(e)}"],                          # List containing error message as response
                [""],                                           # List containing empty negative prompt
                [None],                                         # List containing no OMNI data
                image_tensor,                                  # Single tensor
                mask_tensor                                    # Single tensor
            )

    def create_error_response(self, images, masks, error_message, prompt=""):
        """Create standardized error response"""
        try:
            if images is None:
                image_tensor = load_placeholder_image(self.placeholder_image_path)[0]
            else:
                image_tensor = images
            if masks is None:
                mask_tensor = load_placeholder_image(self.placeholder_image_path)[1]
            else:
                mask_tensor = masks
            return {
                "Question": prompt,
                "Response": f"Error: {error_message}",
                "Negative": f"Error: {error_message}",
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
                "Negative": f"Error: {error_message}",
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
    "IF_LLM": IFLLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_LLM": "IF LLM 🎨"
}
