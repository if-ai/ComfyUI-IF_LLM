# transformers_api.py
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2VLProcessor,
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
    AutoTokenizer,
)
from qwen_vl_utils import process_vision_info
from typing import List, Union, Optional, Dict, Any
from PIL import Image
from io import BytesIO
import base64
import torch
import logging
import os
import re
import gc
import time
from folder_paths import models_dir
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import json
import importlib
import importlib.util 
import comfy.model_management as mm
from torchvision.transforms import functional as TF
import numpy as np
import tempfile
import shutil
import sys
import glob
from importlib.machinery import SourceFileLoader
import traceback

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IF_LLM.transformers_api")

try:
    import psutil
except ImportError:
    psutil = None

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoProcessor,
        StoppingCriteria, 
        StoppingCriteriaList,
        TextIteratorStreamer
    )
except ImportError:
    logging.warning("Transformers not installed, some functionality may be limited")

try:
    # Try to import qwen_vl_utils for better AWQ compatibility
    import qwen_vl_utils
except ImportError:
    logging.warning("qwen_vl_utils not found. Some AWQ models might not load correctly.")

class TransformersModelManager:
    def __init__(self):
        """Initialize the TransformersModelManager"""
        # Set up paths and model configurations
        self.models_dir = models_dir
        self.llm_path = os.path.join(self.models_dir, "LLM")
        
        # Create the LLM directory if it doesn't exist
        os.makedirs(self.llm_path, exist_ok=True)
        
        # Set model load arguments
        self.model_load_args = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }
        
        # Model tracking
        self.loaded_models = {}
        self.current_model_name = None
        self.current_model_type = None
        self.last_model_usage = 0
        
        # Configure model paths
        self.configure_model_paths()
        
    def configure_model_paths(self):
        """Set up paths for different model types"""
        # Models configuration
        self.model_configs = {
            "Qwen/QwQ-32B-AWQ": {
                "model_type": "awq",
                "processor_type": "auto",
                "hf_repo": "Qwen/QwQ-32B-AWQ",
                "local_dir": os.path.join(self.models_dir, "LLM", "QwQ-32B-AWQ"),
                "min_vram_gb": 20,  # This model really needs at least this much VRAM
                "recommended_vram_gb": 24,  # Ideally it would have this much
                "supports_vision": False,
                "context_length": 32768,
                "requires_autoawq": True
            },
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ": {
                "model_type": "qwen2_5_vl",
                "processor_type": "qwen2_5_vl",
                "hf_repo": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                "local_dir": os.path.join(self.models_dir, "LLM", "Qwen2.5-VL-3B-Instruct-AWQ"),
                "min_vram_gb": 4,
                "recommended_vram_gb": 8,
                "supports_vision": True,
                "context_length": 32768,
                "requires_autoawq": True
            },
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ": {
                "model_type": "qwen2_5_vl",
                "processor_type": "qwen2_5_vl",
                "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                "local_dir": os.path.join(self.models_dir, "LLM", "Qwen2.5-VL-7B-Instruct-AWQ"),
                "min_vram_gb": 8,
                "recommended_vram_gb": 12,
                "supports_vision": True,
                "context_length": 32768,
                "requires_autoawq": True
            },
            "Qwen/Qwen2.5-7B-Instruct": {
                "model_type": "auto",
                "processor_type": "auto",
                "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
                "local_dir": os.path.join(self.models_dir, "LLM", "Qwen2.5-7B-Instruct"),
                "min_vram_gb": 14,
                "recommended_vram_gb": 16,
                "supports_vision": False,
                "context_length": 32768,
                "requires_autoawq": False
            },
            "Qwen/Qwen2.5-VL-7B-Instruct": {
                "model_type": "qwen2_5_vl",
                "processor_type": "qwen2_5_vl",
                "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
                "local_dir": os.path.join(self.models_dir, "LLM", "Qwen2.5-VL-7B-Instruct"),
                "min_vram_gb": 16,
                "recommended_vram_gb": 24,
                "supports_vision": True,
                "context_length": 32768,
                "requires_autoawq": False
            },
            "Qwen/Qwen2.5-VL-3B-Instruct": {
                "model_type": "qwen2_5_vl",
                "processor_type": "qwen2_5_vl",
                "hf_repo": "Qwen/Qwen2.5-VL-3B-Instruct",
                "local_dir": os.path.join(self.models_dir, "LLM", "Qwen2.5-VL-3B-Instruct"),
                "min_vram_gb": 6,
                "recommended_vram_gb": 12,
                "supports_vision": True,
                "context_length": 32768,
                "requires_autoawq": False
            },
            # Add other models as needed
        }
        
    def clean_memory(self):
        """Clean up memory to prepare for model loading or after use"""
        try:
            import gc
            import torch
            import logging
            import time
            start_time = time.time()
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # Log current VRAM usage for each GPU
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # Convert to GB
                    logging.info(f"GPU {i}: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")
            
            # Log RAM usage
            try:
                import psutil
                ram_percent = psutil.virtual_memory().percent
                logging.info(f"RAM usage: {ram_percent:.1f}%")
            except ImportError:
                logging.info("psutil not available, skipping RAM usage reporting")
                
            elapsed = time.time() - start_time
            logging.info(f"Memory cleaned in {elapsed:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error cleaning memory: {str(e)}")
            
        return True
    
    def ensure_model_downloaded(self, model_name):
        """Make sure the model is downloaded and available locally"""
        try:
            if model_name not in self.model_configs:
                logging.error(f"Model {model_name} not found in configurations")
                return False
                
            model_config = self.model_configs[model_name]
            local_dir = model_config["local_dir"]
            hf_repo = model_config["hf_repo"]
            
            logging.info(f"Checking for model {model_name} at {local_dir}")
            
            # Check if the model exists locally
            if os.path.exists(local_dir) and os.path.isdir(local_dir):
                logging.info(f"Model {model_name} found locally at {local_dir}")
                
                # Check for config.json to verify it's a valid model directory
                if os.path.exists(os.path.join(local_dir, "config.json")):
                    logging.info(f"Model {model_name} directory contains config.json, proceeding")
                    return True
                else:
                    logging.warning(f"Model directory for {model_name} exists but may be incomplete (no config.json)")
                    # Continue to download to be safe
            
            # Create the destination directory if it doesn't exist
            os.makedirs(local_dir, exist_ok=True)
            
            # Download model files from Hugging Face
            logging.info(f"Downloading model {model_name} from {hf_repo} to {local_dir}...")
            
            try:
                # Use snapshot_download to download the full model repository
                hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
                if hf_token:
                    logging.info("Using Hugging Face token from environment")
                else:
                    logging.info("No Hugging Face token found in environment")
                    
                snapshot_download(
                    repo_id=hf_repo,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=hf_token
                )
                
                logging.info(f"Model {model_name} downloaded successfully to {local_dir}")
                return True
                
            except Exception as dl_error:
                logging.error(f"Error downloading model {model_name}: {str(dl_error)}")
                import traceback
                logging.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logging.error(f"Error ensuring model download for {model_name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
            
    def load_model(self, model_name):
        """Load a model and its processor by name"""
        
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
            import gc
            import logging
            
            logging.info(f"Attempting to load model: {model_name}")
            
            # Clean up memory before loading new model
            self.clean_memory()
            
            # Check if model_name is in our configurations
            if model_name not in self.model_configs:
                logging.error(f"Model {model_name} not found in configurations. Available models: {list(self.model_configs.keys())}")
                
                # Try fallback to direct model loading from Hugging Face
                logging.info(f"Attempting fallback to direct model loading for: {model_name}")
                try:
                    # For AWQ models
                    if "awq" in model_name.lower():
                        try:
                            # Instead of trying to import autoawq, use transformers directly
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            logging.info("Using transformers for AWQ model loading")
                        except ImportError:
                            logging.warning("Transformers not found. Please install with: pip install transformers")
                            return {"error": "Failed to load model: transformers package is required"}
                            
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto",
                            torch_dtype=torch.float16
                        )
                        processor = AutoTokenizer.from_pretrained(model_name)
                    else:
                        # Regular model
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto",
                            torch_dtype=torch.float16
                        )
                        processor = AutoTokenizer.from_pretrained(model_name)
                        
                    # Store model and processor
                    model_data = {
                        "model": model,
                        "processor": processor,
                        "supports_vision": True,  # Force supports_vision to True for Qwen VL models
                        "context_length": 4096  # Default context length
                    }
                    self.loaded_models[model_name] = model_data
                    logging.info(f"Successfully loaded model directly: {model_name}")
                    return {"status": "success", "message": f"Model {model_name} loaded successfully via fallback"}
                
                except Exception as fallback_error:
                    logging.error(f"Fallback loading failed: {str(fallback_error)}")
                    return {"error": f"Model {model_name} not found in configurations and fallback loading failed"}
            
            model_config = self.model_configs[model_name]
            logging.info(f"Found model configuration: {model_config}")
            
            # Download or find the model locally
            if not self.ensure_model_downloaded(model_name):
                logging.error(f"Failed to download or locate model: {model_name}")
                
                # Try direct loading as a fallback
                try:
                    logging.info(f"Attempting direct loading from Hugging Face: {model_name}")
                    
                    if model_config.get("requires_autoawq", False):
                        try:
                            # Instead of trying to import autoawq, use transformers directly
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            logging.info("Using transformers for AWQ model loading")
                        except ImportError:
                            logging.warning("Transformers not found. Please install with: pip install transformers")
                            return {"error": "Failed to load model: transformers package is required"}
                        
                        if "qwen2_5_vl" in model_config.get("model_type", ""):
                            # Special handling for Qwen VL models - improved approach
                            logging.info(f"Loading Qwen VL AWQ model: {model_name}")
                            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                            
                            # Try to determine the best device configuration
                            device = "cuda:0" if torch.cuda.is_available() else "cpu"
                            logging.info(f"Using device: {device} for Qwen VL model")
                            
                            # Set appropriate dtype
                            dtype = torch.float16
                            
                            # Check if flash attention is available
                            can_use_flash_attn = False
                            if torch.cuda.is_available():
                                try:
                                    from flash_attn import flash_attn_func
                                    can_use_flash_attn = True
                                    logging.info("Flash attention available, will use for better performance")
                                except ImportError:
                                    pass
                            
                            # Setup model loading parameters - similar to VideoPromptsNode
                            model_kwargs = {
                                "torch_dtype": dtype,
                                "trust_remote_code": True,
                            }
                            
                            # Don't use device_map with AWQ models to avoid CPU offloading
                            # Just use the specific device
                            model_kwargs.pop("device_map", None)
                            
                            # Add flash attention if available
                            if can_use_flash_attn:
                                model_kwargs["attn_implementation"] = "flash_attention_2"
                            
                            # Load the model with better settings
                            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                model_config.get("local_dir", model_config["hf_repo"]),
                                **model_kwargs
                            )
                            
                            # Move to specific device to avoid CPU offloading with AWQ
                            if torch.cuda.is_available():
                                model = model.to(device)
                            
                            # For processor, use standard settings as in VideoPromptsNode
                            processor = AutoProcessor.from_pretrained(
                                model_config.get("local_dir", model_config["hf_repo"]),
                                trust_remote_code=True
                            )
                            
                            # Store model and processor
                            model_data = {
                                "model": model,
                                "processor": processor,
                                "supports_vision": True, 
                                "context_length": model_config.get("context_length", 32768)
                            }
                            self.loaded_models[model_name] = model_data
                            logging.info(f"Successfully loaded Qwen VL model: {model_name}")
                            return model, processor
                            
                        elif "qwen2-vl" in model_config.get("model_type", ""):
                            # Similar approach for Qwen2-VL models
                            logging.info(f"Loading Qwen2-VL AWQ model: {model_name}")
                            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                            
                            # Try to determine the best device configuration
                            device = "cuda:0" if torch.cuda.is_available() else "cpu"
                            logging.info(f"Using device: {device} for Qwen VL model")
                            
                            # Set appropriate dtype
                            dtype = torch.float16
                            
                            # Check if flash attention is available
                            can_use_flash_attn = False
                            if torch.cuda.is_available():
                                try:
                                    from flash_attn import flash_attn_func
                                    can_use_flash_attn = True
                                    logging.info("Flash attention available, will use for better performance")
                                except ImportError:
                                    pass
                            
                            # Setup model loading parameters - similar to VideoPromptsNode
                            model_kwargs = {
                                "torch_dtype": dtype,
                                "trust_remote_code": True,
                            }
                            
                            # Don't use device_map with AWQ models to avoid CPU offloading
                            # Just use the specific device
                            model_kwargs.pop("device_map", None)
                            
                            # Add flash attention if available
                            if can_use_flash_attn:
                                model_kwargs["attn_implementation"] = "flash_attention_2"
                            
                            # Load the model with better settings
                            model = Qwen2VLForConditionalGeneration.from_pretrained(
                                model_config.get("local_dir", model_config["hf_repo"]),
                                **model_kwargs
                            )
                            
                            # Move to specific device to avoid CPU offloading with AWQ
                            if torch.cuda.is_available():
                                model = model.to(device)
                            
                            # For processor, use standard settings as in VideoPromptsNode
                            processor = AutoProcessor.from_pretrained(
                                model_config.get("local_dir", model_config["hf_repo"]),
                                trust_remote_code=True
                            )
                            
                            # Store model and processor
                            model_data = {
                                "model": model,
                                "processor": processor,
                                "supports_vision": True,
                                "context_length": model_config.get("context_length", 32768)
                            }
                            self.loaded_models[model_name] = model_data
                            logging.info(f"Successfully loaded Qwen VL model: {model_name}")
                            return model, processor

                    else:
                        # Default model loading if not a special case
                        logging.info(f"Loading model with standard transformers approach: {model_name}")
                        if model_config["model_type"] == "auto":
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map="auto",
                                torch_dtype=torch.float16
                            )
                            processor = AutoTokenizer.from_pretrained(model_path)
                        elif model_config["model_type"] == "vision":
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map="auto",
                                torch_dtype=torch.float16
                            )
                            processor = AutoProcessor.from_pretrained(model_path)
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map="auto",
                                torch_dtype=torch.float16
                            )
                            processor = AutoTokenizer.from_pretrained(model_path)
                            
                        # Store loaded model
                        model_data = {
                            "model": model,
                            "processor": processor,
                            "supports_vision": True,  # Force supports_vision to True for Qwen VL models
                            "context_length": model_config.get("context_length", 4096)
                        }
                        self.loaded_models[model_name] = model_data
                        logging.info(f"Successfully loaded model: {model_name}")
                        
                    return {"status": "success", "message": f"Model {model_name} loaded successfully"}
                    
                except Exception as e:
                    logging.error(f"Error loading model {model_name}: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
                    return {"error": f"Failed to load model: {str(e)}"}
            
            # Use local path if available
            model_path = model_config.get("local_dir", model_config["hf_repo"])
            logging.info(f"Using model path: {model_path}")
            
            # For AWQ models, we need to check and use autoawq
            if model_config.get("requires_autoawq", False):
                try:
                    logging.info(f"Model {model_name} requires AutoAWQ")
                    # Check for autoawq
                    try:
                        # Instead of trying to import autoawq, use transformers directly
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        logging.info("Using transformers for AWQ model loading")
                    except ImportError:
                        logging.warning("Transformers not found. Please install with: pip install transformers")
                        return {"error": "Failed to load model: transformers package is required"}
                    
                    if "qwen2_5_vl" in model_config.get("model_type", ""):
                        # Special handling for Qwen VL models
                        logging.info(f"Loading Qwen VL AWQ model: {model_name}")
                        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
                        
                        # For Qwen models use specific device mapping and dtype
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            model_path,
                            device_map="cuda:0",
                            torch_dtype=torch.float16,
                            trust_remote_code=True
                        )
                        
                        # Store model and processor
                        model_data = {
                            "model": model,
                            "processor": tokenizer,
                            "supports_vision": True,  # Force supports_vision to True for Qwen VL models
                            "context_length": model_config.get("context_length", 4096)
                        }
                        self.loaded_models[model_name] = model_data
                        logging.info(f"Successfully loaded AWQ model: {model_name}")
                        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
                    else:
                        # For other AWQ models
                        logging.info(f"Loading standard AWQ model: {model_name}")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map="auto",
                            torch_dtype=torch.float16
                        )
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        
                        # Store model and processor
                        model_data = {
                            "model": model,
                            "processor": tokenizer,
                            "supports_vision": True,  # Force supports_vision to True for Qwen VL models
                            "context_length": model_config.get("context_length", 4096)
                        }
                        self.loaded_models[model_name] = model_data
                        logging.info(f"Successfully loaded AWQ model: {model_name}")
                        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
                        
                except Exception as e:
                    logging.error(f"Error loading AWQ model {model_name}: {str(e)}")
                    return {"error": f"Failed to load AWQ model: {str(e)}"}
            
            # Default model loading if not a special case
            logging.info(f"Loading model with standard transformers approach: {model_name}")
            if model_config["model_type"] == "auto":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                processor = AutoTokenizer.from_pretrained(model_path)
            elif model_config["model_type"] == "vision":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                processor = AutoProcessor.from_pretrained(model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                processor = AutoTokenizer.from_pretrained(model_path)
                
            # Store loaded model
            model_data = {
                "model": model,
                "processor": processor,
                "supports_vision": True,  # Force supports_vision to True for Qwen VL models
                "context_length": model_config.get("context_length", 4096)
            }
            self.loaded_models[model_name] = model_data
            logging.info(f"Successfully loaded model: {model_name}")
            
            return {"status": "success", "message": f"Model {model_name} loaded successfully"}
            
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {"error": f"Failed to load model: {str(e)}"}

    def process_frames(self, images_tensor, frame_sample_count=16, max_pixels=512*512):
        """
        Process image frames for the model by sampling and resizing.
        Similar to IF_VideoPromptsNode's implementation.
        
        Args:
            images_tensor: Input tensor in shape [B,H,W,C]
            frame_sample_count: Number of frames to sample
            max_pixels: Max pixels for each frame
            
        Returns:
            List of processed PIL images
        """
        # Import required libraries
        import math
        from PIL import Image
        import numpy as np
        
        # Get the batch size (number of frames)
        batch_size = images_tensor.shape[0]
        
        # Sample the frames evenly from the batch
        if batch_size <= frame_sample_count:
            # Use all frames if we have fewer than requested
            sampled_indices = list(range(batch_size))
        else:
            # Sample evenly across the frames
            sampled_indices = [int(i * (batch_size - 1) / (frame_sample_count - 1)) for i in range(frame_sample_count)]
        
        # Extract the sampled frames from the tensor
        sampled_frames = [images_tensor[i] for i in sampled_indices]
        
        # Convert to PIL images
        pil_images = []
        for frame in sampled_frames:
            # Convert tensor to numpy
            frame_np = frame.cpu().numpy()
            # Scale from [0,1] to [0,255]
            frame_np = (frame_np * 255).astype(np.uint8)
            # Convert to PIL
            pil_image = Image.fromarray(frame_np)
            
            # Calculate resize dimensions if needed
            if max_pixels > 0:
                width, height = pil_image.size
                if width * height > max_pixels:
                    scale = math.sqrt(max_pixels / (width * height))
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            pil_images.append(pil_image)
        
        return pil_images

    async def send_transformers_request(
        self,
        model_name: str,
        user_prompt: str,
        system_message: str = "",
        messages: List[Dict[str, Any]] = None,
        images: List[str] = None,
        seed: int = 42,
        random: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop_string: str = "",
        precision: str = "fp16",
        attention: str = "sdpa",
        keep_alive: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a request using the transformers library
        """
        try:
            import torch
            from transformers import TextIteratorStreamer
            import threading
            import numpy as np
            from PIL import Image  # Re-import PIL.Image in this scope to avoid UnboundLocalError
            
            # Set random seed if needed
            if not random and seed != -1:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Load/reload model if needed
            load_result = self.load_model(model_name)
            
            if "error" in load_result:
                return load_result
                
            model_data = self.loaded_models.get(model_name)
            if not model_data:
                return {"error": f"Failed to access model data for {model_name}"}
                
            model = model_data["model"]
            processor = model_data["processor"]
            supports_vision = model_data.get("supports_vision", False)
            
            # Process images if available and model supports vision
            pil_images = []
            if images is not None and supports_vision:
                # Check if images is a tensor, list, or other container with content
                has_images = False
                if isinstance(images, torch.Tensor):
                    has_images = images.nelement() > 0
                elif isinstance(images, list) or isinstance(images, tuple):
                    has_images = len(images) > 0
                else:
                    has_images = bool(images)  # Only use direct boolean conversion for non-tensor types
                
                if has_images:
                    logging.info(f"Processing images for vision model")
                    
                    # Import these here to ensure they're available in this scope
                    from PIL import Image
                    import numpy as np
                    import io
                    import base64
                    
                    # Handle different types of image inputs
                    if isinstance(images, torch.Tensor):
                        # If it's a tensor, convert each image to PIL
                        logging.info(f"Image tensor shape: {images.shape}")
                        
                        # Determine tensor layout
                        if images.dim() == 4:  # [batch, channels, height, width] or [batch, height, width, channels]
                            for i in range(images.shape[0]):
                                img_tensor = images[i]
                                
                                # Check if [C, H, W] or [H, W, C] format
                                if images.shape[1] == 3 or images.shape[1] == 4:  # [B, C, H, W]
                                    # Convert from [C, H, W] to [H, W, C] for PIL
                                    img_tensor = img_tensor.permute(1, 2, 0)
                                    logging.info(f"Processing BCHW tensor, image {i}")
                                # else assume [B, H, W, C] format, which needs no permutation
                                
                                # Ensure values are in 0-255 range and convert to uint8
                                img_np = img_tensor.cpu().numpy()
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255).astype(np.uint8)
                                else:
                                    img_np = img_np.astype(np.uint8)
                                
                                # Convert to PIL
                                pil_img = Image.fromarray(img_np)
                                pil_images.append(pil_img)
                                logging.info(f"Processed tensor image {i}: {pil_img.size}")
                        elif images.dim() == 3:  # Single image [C, H, W] or [H, W, C]
                            img_tensor = images
                            # Check if [C, H, W] format and convert to [H, W, C]
                            if img_tensor.shape[0] == 3 or img_tensor.shape[0] == 4:
                                img_tensor = img_tensor.permute(1, 2, 0)
                                logging.info(f"Processing CHW tensor")
                            # else assume [H, W, C] format, which needs no permutation
                            
                            # Ensure values are in 0-255 range and convert to uint8
                            img_np = img_tensor.cpu().numpy()
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype(np.uint8)
                            else:
                                img_np = img_np.astype(np.uint8)
                            
                            # Convert to PIL
                            pil_img = Image.fromarray(img_np)
                            pil_images.append(pil_img)
                            logging.info(f"Processed single tensor image: {pil_img.size}")
                    else:
                        # Original handling for string/base64 inputs or PIL images
                        for img_item in images:
                            try:
                                # Check if it's a numpy array
                                if isinstance(img_item, np.ndarray):
                                    pil_image = Image.fromarray(img_item)
                                    pil_images.append(pil_image)
                                    logging.info(f"Processed numpy image: {pil_image.size}")
                                # Check if it's a base64 string
                                elif isinstance(img_item, str) and img_item.startswith("data:image"):
                                    # Extract the base64 data
                                    img_data = img_item.split(",")[1]
                                    pil_image = Image.open(io.BytesIO(base64.b64decode(img_data)))
                                    pil_images.append(pil_image)
                                    logging.info(f"Processed base64 image: {pil_image.size}")
                                elif isinstance(img_item, Image.Image):
                                    # Already a PIL image
                                    pil_images.append(img_item)
                                    logging.info(f"Processed PIL image: {img_item.size}")
                                # Handle torch tensor in list
                                elif isinstance(img_item, torch.Tensor):
                                    if img_item.dim() == 3:  # [C, H, W] or [H, W, C]
                                        # Check if [C, H, W] format and convert to [H, W, C]
                                        if img_item.shape[0] == 3 or img_item.shape[0] == 4:
                                            img_tensor = img_item.permute(1, 2, 0)
                                        else:
                                            img_tensor = img_item
                                        
                                        # Ensure values are in 0-255 range and convert to uint8
                                        img_np = img_tensor.cpu().numpy()
                                        if img_np.max() <= 1.0:
                                            img_np = (img_np * 255).astype(np.uint8)
                                        else:
                                            img_np = img_np.astype(np.uint8)
                                        
                                        # Convert to PIL
                                        pil_img = Image.fromarray(img_np)
                                        pil_images.append(pil_img)
                                        logging.info(f"Processed tensor from list: {pil_img.size}")
                                else:
                                    logging.warning(f"Unsupported image format: {type(img_item)}")
                            except Exception as e:
                                logging.error(f"Error processing image: {e}")
                                continue
            
            # Construct conversation messages
            chat_messages = self.construct_messages(model_name, system_message, user_prompt, messages, pil_images)
            
            # Generate model inputs
            if "qwen2.5-vl" in model_name.lower() and pil_images:
                try:
                    logging.info(f"Using approach inspired by VideoPromptsNode for Qwen2.5-VL")
                    
                    # Memory cleanup
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Force model to CUDA:0 for ComfyUI compatibility
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    
                    # First, ensure we have the correct processor
                    from transformers import AutoProcessor
                    
                    # Make sure we explicitly create the processor from the model path
                    processor = AutoProcessor.from_pretrained(
                        model.name_or_path,
                        trust_remote_code=True
                    )
                    
                    # Prepare the prompt text
                    prompt_text = "Analyze this image and describe what you see in detail."
                    if system_message:
                        # Add the system message to instruct the model
                        system_content = system_message
                    else:
                        system_content = "You are a helpful assistant that analyzes images and provides detailed descriptions."
                    
                    # If there's a user prompt, use it instead of the default
                    if user_prompt and user_prompt.strip():
                        prompt_text = user_prompt
                    
                    # Handle multiple frames/images and limit to a reasonable number
                    # If we have more than 16 images, sample them evenly
                    max_frame_count = 16  # Limit to 16 frames to avoid VRAM issues
                    
                    if len(pil_images) > max_frame_count:
                        logging.info(f"Found {len(pil_images)} images, sampling down to {max_frame_count}")
                        
                        # Convert pil_images to tensor format if needed for processing
                        if hasattr(pil_images, 'shape') and len(pil_images.shape) == 4:
                            # Already a tensor
                            processed_images = self.process_frames(pil_images, max_frame_count)
                        else:
                            # Already list of PIL images, sample manually
                            if len(pil_images) <= max_frame_count:
                                processed_images = pil_images
                            else:
                                # Sample evenly
                                indices = [int(i * (len(pil_images) - 1) / (max_frame_count - 1)) for i in range(max_frame_count)]
                                processed_images = [pil_images[i] for i in indices]
                    else:
                        processed_images = pil_images
                    
                    # Format as a chat with proper message structure - this is key
                    # This is how the VideoPromptsNode handles it
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": []}
                    ]
                    
                    # Add images to the user message - this approach is crucial
                    for img in processed_images:
                        messages[1]["content"].append({"type": "image", "image": img})
                    
                    # Add the text part of the user message
                    messages[1]["content"].append({"type": "text", "text": prompt_text})
                    
                    # Use apply_chat_template to format messages - just like VideoPromptsNode
                    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    logging.info(f"Created chat template with length: {len(chat_text)}")
                    
                    # Extract images for processing
                    image_inputs = [item["image"] for item in messages[1]["content"] if "type" in item and item["type"] == "image"]
                    
                    # Process exactly like VideoPromptsNode
                    inputs = processor(text=[chat_text], images=image_inputs, return_tensors="pt")
                    
                    # Move to the right device
                    inputs = inputs.to(device)
                    model.to(device)
                    
                    # Log input shapes for debugging
                    logging.info(f"Input keys: {inputs.keys()}")
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            logging.info(f"{key} shape: {inputs[key].shape}, device: {inputs[key].device}")
                    
                    # Set seed if needed
                    if not random and seed is not None:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                    
                    # Generate with parameters directly without all the special processing
                    logging.info(f"Starting model.generate()")
                    
                    # Use the exact same generation approach as VideoPromptsNode
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            do_sample=random or temperature > 0.01,
                            temperature=max(0.01, temperature),
                            max_new_tokens=max_tokens,
                            top_k=top_k if random else 50,
                            top_p=top_p if random else 1.0,
                            repetition_penalty=repeat_penalty if repeat_penalty > 1.0 else None
                        )
                    
                    # Decode exactly like VideoPromptsNode
                    generated_text = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )[0]
                    
                    logging.info(f"Generated response of length {len(generated_text)}")
                    
                    # Clean up memory explicitly
                    del inputs
                    del outputs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Clean up if not keeping the model alive
                    if not keep_alive:
                        self.unload_model(model_name)
                        
                    return {
                        "status": "success",
                        "response": generated_text,
                        "model": model_name
                    }
                except Exception as gen_error:
                    logging.error(f"Error during generation: {gen_error}")
                    import traceback
                    logging.error(traceback.format_exc())
                    # Clean up memory on error
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    return {"error": f"Failed to generate response with Qwen2.5-VL: {str(gen_error)}"}

            elif "qwen2-vl" in model_name.lower() and pil_images:
                try:
                    logging.info(f"Using approach inspired by VideoPromptsNode for Qwen2-VL")
                    
                    # Memory cleanup
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Force model to CUDA:0 for ComfyUI compatibility
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    
                    # First, ensure we have the correct processor
                    from transformers import AutoProcessor
                    
                    # Make sure we explicitly create the processor from the model path
                    processor = AutoProcessor.from_pretrained(
                        model.name_or_path,
                        trust_remote_code=True
                    )
                    
                    # Prepare the prompt text
                    prompt_text = "Analyze this image and describe what you see in detail."
                    if system_message:
                        # Add the system message to instruct the model
                        system_content = system_message
                    else:
                        system_content = "You are a helpful assistant that analyzes images and provides detailed descriptions."
                    
                    # If there's a user prompt, use it instead of the default
                    if user_prompt and user_prompt.strip():
                        prompt_text = user_prompt
                    
                    # Handle multiple frames/images and limit to a reasonable number
                    # If we have more than 16 images, sample them evenly
                    max_frame_count = 16  # Limit to 16 frames to avoid VRAM issues
                    
                    if len(pil_images) > max_frame_count:
                        logging.info(f"Found {len(pil_images)} images, sampling down to {max_frame_count}")
                        
                        # Convert pil_images to tensor format if needed for processing
                        if hasattr(pil_images, 'shape') and len(pil_images.shape) == 4:
                            # Already a tensor
                            processed_images = self.process_frames(pil_images, max_frame_count)
                        else:
                            # Already list of PIL images, sample manually
                            if len(pil_images) <= max_frame_count:
                                processed_images = pil_images
                            else:
                                # Sample evenly
                                indices = [int(i * (len(pil_images) - 1) / (max_frame_count - 1)) for i in range(max_frame_count)]
                                processed_images = [pil_images[i] for i in indices]
                    else:
                        processed_images = pil_images
                    
                    # Format as a chat with proper message structure - this is key
                    # This is how the VideoPromptsNode handles it
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": []}
                    ]
                    
                    # Add images to the user message - this approach is crucial
                    for img in processed_images:
                        messages[1]["content"].append({"type": "image", "image": img})
                    
                    # Add the text part of the user message
                    messages[1]["content"].append({"type": "text", "text": prompt_text})
                    
                    # Use apply_chat_template to format messages - just like VideoPromptsNode
                    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    logging.info(f"Created chat template with length: {len(chat_text)}")
                    
                    # Extract images for processing
                    image_inputs = [item["image"] for item in messages[1]["content"] if "type" in item and item["type"] == "image"]
                    
                    # Process exactly like VideoPromptsNode
                    inputs = processor(text=[chat_text], images=image_inputs, return_tensors="pt")
                    
                    # Move to the right device
                    inputs = inputs.to(device)
                    model.to(device)
                    
                    # Log input shapes for debugging
                    logging.info(f"Input keys: {inputs.keys()}")
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            logging.info(f"{key} shape: {inputs[key].shape}, device: {inputs[key].device}")
                    
                    # Set seed if needed
                    if not random and seed is not None:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                    
                    # Generate with parameters directly without all the special processing
                    logging.info(f"Starting model.generate()")
                    
                    # Use the exact same generation approach as VideoPromptsNode
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            do_sample=random or temperature > 0.01,
                            temperature=max(0.01, temperature),
                            max_new_tokens=max_tokens,
                            top_k=top_k if random else 50,
                            top_p=top_p if random else 1.0,
                            repetition_penalty=repeat_penalty if repeat_penalty > 1.0 else None
                        )
                    
                    # Decode exactly like VideoPromptsNode
                    generated_text = processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )[0]
                    
                    logging.info(f"Generated response of length {len(generated_text)}")
                    
                    # Clean up memory explicitly
                    del inputs
                    del outputs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Clean up if not keeping the model alive
                    if not keep_alive:
                        self.unload_model(model_name)
                        
                    return {
                        "status": "success",
                        "response": generated_text,
                        "model": model_name
                    }
                except Exception as gen_error:
                    logging.error(f"Error during generation: {gen_error}")
                    import traceback
                    logging.error(traceback.format_exc())
                    # Clean up memory on error
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    return {"error": f"Failed to generate response with Qwen2-VL: {str(gen_error)}"}

            elif hasattr(processor, "apply_chat_template"):
                # Use chat template for text generation
                text = processor.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = processor([text], return_tensors="pt").to(model.device)
            else:
                # Fallback for models without chat template
                if processor.pad_token is None:
                    processor.pad_token = processor.eos_token
                
                # Add system message if provided
                if system_message:
                    prompt = f"{system_message}\n\n{user_prompt}"
                else:
                    prompt = user_prompt
                    
                model_inputs = processor(prompt, return_tensors="pt").to(model.device)
            
            # Set up generation parameters
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": random or temperature > 0.01,
                "temperature": max(0.01, temperature) if random else 1.0,
            }
            
            if random and top_p < 1.0:
                generation_kwargs["top_p"] = top_p
                
            if random and top_k > 0:
                generation_kwargs["top_k"] = top_k
                
            if repeat_penalty > 1.0:
                generation_kwargs["repetition_penalty"] = repeat_penalty
                
            if stop_string:
                stopping_criteria = self.create_stopping_criteria(processor, stop_string)
                if stopping_criteria:
                    generation_kwargs["stopping_criteria"] = stopping_criteria
            
            # Set up streaming for non-blocking generation
            streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer
            
            # Generate text in a separate thread
            generated_text = []
            
            # Prepare kwargs for model.generate
            model_generate_kwargs = {}
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    # Ensure all tensors are on the same device as the model
                    model_generate_kwargs[k] = v.to(model.device)
                else:
                    model_generate_kwargs[k] = v
                
            # Add generation config
            for k, v in generation_kwargs.items():
                model_generate_kwargs[k] = v
            
            # Create thread for generation
            try:
                thread = threading.Thread(
                    target=model.generate,
                    kwargs=model_generate_kwargs
                )
                thread.start()
                
                # Collect and process streamed output
                for text_chunk in streamer:
                    generated_text.append(text_chunk)
                    
                thread.join(timeout=60)  # Set a timeout to prevent hanging
                if thread.is_alive():
                    logging.warning("Generation thread is taking too long, may be stuck")
                    return {"error": "Model generation timeout"}
                    
                response = "".join(generated_text)
                
                # Additional processing for Qwen models
                if "qwen2.5-vl" in model_name.lower() or "qwen2-vl" in model_name.lower():
                    # Clean up Qwen model responses
                    response = response.replace("<|im_end|>", "")
                    response = self.post_process_response(response)
                    logging.info(f"Processed Qwen-VL response: {response[:100]}...")
                else:
                    # Regular post-processing for other models
                    response = self.post_process_response(response)
                    
                # Clean up if not keeping the model alive
                if not keep_alive:
                    self.unload_model(model_name)
                    
                return {
                    "status": "success",
                    "response": response,
                    "model": model_name
                }
            except Exception as gen_error:
                logging.error(f"Error during model generation: {str(gen_error)}")
                import traceback
                logging.error(traceback.format_exc())
                return {"error": f"Failed during generation: {str(gen_error)}"}
            
        except Exception as e:
            logging.error(f"Error in transformers request: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {"error": f"Failed to process request: {str(e)}"}

    def post_process_response(self, response):
        pattern = r'^(###\s*)?(?:Assistant|AI):\s*'
        response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        response = response.lstrip()
        response = re.sub(r'\n(###\s*)?(?:Human|User):\s*$', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\n\s*\n', '\n\n', response)
        return response.strip()

    def construct_messages(self, model_name, system_message, user_message, messages, pil_images):
        """Construct properly formatted messages for the model"""
        
        # Default system message if none provided
        if not system_message or system_message.strip() == "":
            system_message = "You are a helpful AI assistant."
            
        # Start with system message
        chat_messages = [{"role": "system", "content": system_message}]
        
        # Add previous messages if provided
        if messages and isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    chat_messages.append(msg)
        
        # Handle image-enabled messages for multimodal models
        if ("qwen2.5-vl" in model_name.lower() or "qwen2-vl" in model_name.lower()) and pil_images:
            logging.info(f"Creating multimodal message with {len(pil_images)} images for Qwen VL model")
            
            # For Qwen VL models, we need to format the user message with images
            if len(pil_images) > 0:
                if isinstance(user_message, str):
                    # Create content list with images and text
                    content = []
                    
                    # Add images first for Qwen VL models
                    for img in pil_images:
                        content.append({"type": "image", "image": img})
                    
                    # Add text last
                    content.append({"type": "text", "text": user_message})
                    
                    # Add user message with images
                    chat_messages.append({"role": "user", "content": content})
                    logging.info(f"Created user message with {len(content)-1} images and text")
                else:
                    # Fallback if user_message is not a string
                    chat_messages.append({"role": "user", "content": [
                        {"type": "image", "image": pil_images[0]},
                        {"type": "text", "text": "Describe this image in detail."}
                    ]})
                    logging.info("Created fallback message with 1 image and default text")
            else:
                # No images, just add text
                chat_messages.append({"role": "user", "content": user_message})
                logging.info("No images provided, using text-only message")
        else:
            # For text-only models or non-Qwen models
            chat_messages.append({"role": "user", "content": user_message})
            
        return chat_messages

    def clean_results(self, results, task):
        if task == 'ocr_with_region':
            clean_results = re.sub(r'</?s>|<[^>]*>', '\n', results)
            clean_results = re.sub(r'\n+', '\n', clean_results)
        else:
            clean_results = results.replace('</s>', '').replace('<s>', '')
        return clean_results

    def unload_model(self, model_name: str):
        """Unload a model to free up memory"""
        if model_name in self.loaded_models:
            logger.info(f"Unloading model {model_name} from memory")
            
            try:
                # Clear references to model and processor
                self.loaded_models[model_name] = None
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info(f"Model {model_name} unloaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {str(e)}")
                return False
        else:
            logger.warning(f"Model {model_name} not found in loaded models, nothing to unload")
            return False

    @classmethod
    def fixed_get_imports(cls, filename: Union[str, os.PathLike], *args, **kwargs) -> List[str]:
        """Remove 'flash_attn' from imports if present."""
        try:
            if not str(filename).endswith("modeling_florence2.py") or not str(filename).endswith("modeling_deepseek.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            if "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports
        except Exception as e:
            print(f"No flash_attn import to remove: {e}")
            return get_imports(filename)

    def create_stopping_criteria(self, tokenizer, stop_string):
        """Create stopping criteria for generation based on a stop string"""
        if not stop_string or not stop_string.strip():
            return None
            
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopStringCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_string, prompt_length=0):
                    self.tokenizer = tokenizer
                    self.stop_string = stop_string
                    self.stop_tokens = tokenizer.encode(stop_string, add_special_tokens=False)
                    self.prompt_length = prompt_length
                    self.current_text = ""
                    
                def __call__(self, input_ids, scores, **kwargs):
                    # Decode the current generation
                    if hasattr(input_ids, 'shape') and len(input_ids.shape) > 1:
                        # For batch inputs, use the first one
                        input_ids_to_decode = input_ids[0][self.prompt_length:]
                    else:
                        # For non-batch inputs
                        input_ids_to_decode = input_ids[self.prompt_length:]
                        
                    # Check if there's anything to decode
                    if len(input_ids_to_decode) == 0:
                        return False
                        
                    new_text = self.tokenizer.decode(input_ids_to_decode, skip_special_tokens=True)
                    self.current_text = new_text
                    
                    # Check if stop string appears in the generated text
                    if self.stop_string in self.current_text:
                        return True
                    return False
            
            return StoppingCriteriaList([StopStringCriteria(tokenizer, stop_string)])
            
        except Exception as e:
            logging.warning(f"Could not create stopping criteria: {str(e)}")
            return None


# Initialize a global manager instance
_transformers_manager = TransformersModelManager()
