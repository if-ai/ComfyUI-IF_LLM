# huggingface_api.py

import aiohttp
import json
import base64
import logging
from typing import List, Union, Optional, Dict, Any, Tuple
from huggingface_hub import InferenceClient
from PIL import Image
import io

logger = logging.getLogger(__name__)

def validate_huggingface_token(api_key: str) -> bool:
    """Validate HuggingFace API token format"""
    if not api_key:
        return False
    # Basic format validation - HF tokens are typically 32-40 characters
    return len(api_key.strip()) >= 32

def get_huggingface_url(model: str) -> str:
    """Format the endpoint URL based on model type"""
    base_url = "https://api-inference.huggingface.co/models/"
    return f"{base_url}{model}"

async def handle_image_generation(
    api_url: str,
    headers: Dict[str, str],
    prompt: str,
    batch_count: int,
    seed: Optional[int],
    base64_images: Optional[List[str]] = None,
    # Image generation specific parameters
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    clip_skip: int = 1,
    control_scale: float = 1.0,
    scheduler: str = "DPMSolverMultistep",
    prompt_2: Optional[str] = None,  # For SDXL models
    negative_prompt_2: Optional[str] = None,  # For SDXL models
    style_preset: Optional[str] = None,  # For SDXL/SD3 models
    target_size: Optional[int] = None,  # For SD3 models
    aesthetic_score: float = 6.0,  # For SDXL models
    original_width: Optional[int] = None,  # For img2img
    original_height: Optional[int] = None,  # For img2img
    strength: float = 0.75,  # For img2img
) -> Dict[str, Any]:
    """Handle text-to-image and image-to-image generation with full parameter control"""
    
    # Determine if we're doing txt2img or img2img
    is_img2img = base64_images is not None and len(base64_images) > 0

    # Base parameters for all models
    parameters = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": batch_count,
        "scheduler": scheduler,
        "clip_skip": clip_skip
    }

    # Add seed if specified
    if seed is not None:
        parameters["seed"] = seed

    # Handle model-specific parameters
    if "stable-diffusion-xl" in api_url.lower() or "sdxl" in api_url.lower():
        # SDXL specific parameters
        parameters.update({
            "width": width,
            "height": height,
            "prompt_2": prompt_2,
            "negative_prompt_2": negative_prompt_2,
            "aesthetic_score": aesthetic_score
        })
        if style_preset:
            parameters["style_preset"] = style_preset
            
    elif "stable-diffusion-3" in api_url.lower() or "sd3" in api_url.lower():
        # SD3 specific parameters
        if target_size:
            parameters["target_size"] = target_size
        if style_preset:
            parameters["style_preset"] = style_preset
    else:
        # Standard SD parameters
        parameters.update({
            "width": width,
            "height": height
        })

    # Handle img2img specific parameters
    if is_img2img:
        parameters.update({
            "strength": strength
        })
        if original_width and original_height:
            parameters["original_width"] = original_width
            parameters["original_height"] = original_height

    # Prepare payload
    if is_img2img:
        payload = {
            "inputs": {
                "prompt": prompt,
                "image": base64_images[0],  # Use first image
                "negative_prompt": negative_prompt,
                **parameters
            }
        }
    else:
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }

    async with aiohttp.ClientSession() as session:
        response = await make_request(session, api_url, headers, payload)
        
        # Handle different response formats
        images = []
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and "image" in item:
                    images.append(item["image"])
                elif isinstance(item, str):
                    images.append(item)
        elif isinstance(response, dict) and "image" in response:
            images.append(response["image"])
        elif isinstance(response, str):
            images.append(response)

        return {"images": images}

async def handle_image_editing(
    api_url: str,
    headers: Dict[str, str],
    prompt: str,
    image: Optional[str],
    mask: Optional[str],
    batch_count: int,
    seed: Optional[int],
    # Image editing specific parameters
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
    scheduler: str = "DPMSolverMultistep",
    control_scale: float = 1.0,
    control_start: float = 0.0,
    control_end: float = 1.0,
    controlnet_conditioning_scale: float = 1.0,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
) -> Dict[str, Any]:
    """Handle image editing operations with full parameter control"""
    
    if not image:
        raise ValueError("Image is required for editing")

    parameters = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "num_images_per_prompt": batch_count,
        "scheduler": scheduler,
        "width": width,
        "height": height,
        "control_scale": control_scale,
        "control_start": control_start,
        "control_end": control_end,
        "controlnet_conditioning_scale": controlnet_conditioning_scale
    }

    if seed is not None:
        parameters["seed"] = seed
    
    if original_width and original_height:
        parameters.update({
            "original_width": original_width,
            "original_height": original_height
        })

    payload = {
        "inputs": {
            "image": image,
            "prompt": prompt,
            **parameters
        }
    }

    if mask:
        payload["inputs"]["mask"] = mask

    async with aiohttp.ClientSession() as session:
        response = await make_request(session, api_url, headers, payload)
        
        # Handle response
        images = []
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and "image" in item:
                    images.append(item["image"])
                elif isinstance(item, str):
                    images.append(item)
        elif isinstance(response, dict) and "image" in response:
            images.append(response["image"])
        elif isinstance(response, str):
            images.append(response)

        return {"images": images}

# Update the main send_huggingface_request function to include these parameters:
async def send_huggingface_request(
    base64_images: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    strategy: str = "normal",
    batch_count: int = 1,
    seed: Optional[int] = None,
    # Basic parameters
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    top_k: int = 40,
    # Image generation parameters
    width: int = 1024,
    height: int = 1024,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
    scheduler: str = "DPMSolverMultistep",
    clip_skip: int = 1,
    control_scale: float = 1.0,
    control_start: float = 0.0,
    control_end: float = 1.0,
    controlnet_conditioning_scale: float = 1.0,
    # SDXL specific
    prompt_2: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    style_preset: Optional[str] = None,
    aesthetic_score: float = 6.0,
    # Other parameters
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None,
    mask: Optional[str] = None,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
) -> Union[Dict[str, Any], str]:
    """Send request to HuggingFace with different strategies and full parameter control"""
    
    try:
        # ... (existing header and URL setup)

        if strategy == "normal":
            return await handle_normal_inference(
                api_url=api_url,
                headers=headers,
                base64_images=base64_images,
                user_message=user_message,
                system_message=system_message,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
            
        elif strategy == "create":
            return await handle_image_generation(
                api_url=api_url,
                headers=headers,
                prompt=user_message,
                batch_count=batch_count,
                seed=seed,
                base64_images=base64_images,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                clip_skip=clip_skip,
                control_scale=control_scale,
                scheduler=scheduler,
                prompt_2=prompt_2,
                negative_prompt_2=negative_prompt_2,
                style_preset=style_preset,
                aesthetic_score=aesthetic_score,
                original_width=original_width,
                original_height=original_height,
                strength=strength
            )
            
        elif strategy == "edit":
            return await handle_image_editing(
                api_url=api_url,
                headers=headers,
                prompt=user_message,
                image=base64_images[0] if base64_images else None,
                mask=mask,
                batch_count=batch_count,
                seed=seed,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                scheduler=scheduler,
                control_scale=control_scale,
                control_start=control_start,
                control_end=control_end,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                original_width=original_width,
                original_height=original_height
            )
            
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    except Exception as e:
        error_msg = f"Error in HuggingFace request: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}