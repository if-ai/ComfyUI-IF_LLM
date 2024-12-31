# HF_api.py
import aiohttp
import base64
import logging
import json
from typing import List, Union, Optional, Dict, Any
from huggingface_hub import InferenceClient
from io import BytesIO
import requests
from PIL import Image

logger = logging.getLogger(__name__)

async def send_huggingface_request(
    base_ip: str, 
    base64_images: Optional[List[str]], 
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.95,
    strategy: str = "normal",
    batch_count: int = 1,
    mask: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Send request to HuggingFace Inference API with support for different strategies
    """
    try:
        if not api_key:
            raise ValueError("HuggingFace API key is required")

        if strategy == "create":
            # Handle text-to-image generation
            return await generate_images(
                model=model,
                prompt=user_message,
                api_key=api_key,
                num_images=batch_count,
                seed=seed,
                negative_prompt=kwargs.get('neg_content', '')
            )
        
        elif strategy == "edit":
            # Handle image-to-image editing
            if not base64_images:
                raise ValueError("Image required for edit strategy")
            
            return await edit_images(
                model=model,
                image=base64_images[0],
                mask=mask,
                prompt=user_message,
                api_key=api_key,
                num_images=batch_count,
                negative_prompt=kwargs.get('neg_content', '')
            )
            
        else: 
            # Handle regular chat/vision requests
            client = InferenceClient(api_key=api_key)
            
            # Prepare messages for VLM
            formatted_messages = prepare_messages(
                system_message=system_message,
                user_message=user_message,
                messages=messages,
                base64_images=base64_images
            )

            response = await run_inference(
                client=client,
                model=model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed
            )

            return format_response(response)

    except Exception as e:
        error_msg = f"Error in HuggingFace API request: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

async def generate_images(
    model: str,
    prompt: str,
    api_key: str,
    num_images: int = 1,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Generate images using HuggingFace text-to-image models"""
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "num_images_per_prompt": num_images,
        }
    }
    
    if seed is not None:
        payload["parameters"]["seed"] = seed

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                # Handle both single image and batch responses
                images = []
                if response.content_type == 'application/json':
                    data = await response.json()
                    images = [d.get("image", "") for d in data]
                else:
                    # Single image as bytes
                    image_bytes = await response.read()
                    images = [base64.b64encode(image_bytes).decode('utf-8')]

                return {
                    "images": images
                }

    except Exception as e:
        logger.error(f"Error generating images: {str(e)}")
        raise

async def edit_images(
    model: str,
    image: str,
    mask: Optional[str],
    prompt: str,
    api_key: str,
    num_images: int = 1,
    negative_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Edit images using HuggingFace image-to-image models"""
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Prepare payload
    payload = {
        "inputs": {
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "num_images": num_images,
        }
    }
    
    if mask is not None:
        payload["inputs"]["mask"] = mask

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                images = []
                if response.content_type == 'application/json':
                    data = await response.json()
                    images = [d.get("image", "") for d in data]
                else:
                    image_bytes = await response.read()
                    images = [base64.b64encode(image_bytes).decode('utf-8')]

                return {
                    "images": images
                }

    except Exception as e:
        logger.error(f"Error editing images: {str(e)}")
        raise

def prepare_messages(
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    base64_images: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Prepare messages for HuggingFace VLM models"""
    prepared_messages = []
    
    if system_message:
        prepared_messages.append({
            "role": "system",
            "content": system_message
        })

    # Add previous messages
    prepared_messages.extend(messages)

    # Add current message with images if present
    if base64_images:
        content = [{"type": "text", "text": user_message}]
        for img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
            })
        prepared_messages.append({"role": "user", "content": content})
    else:
        prepared_messages.append({"role": "user", "content": user_message})

    return prepared_messages

async def run_inference(
    client: InferenceClient,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int] = None
) -> Any:
    """Run inference using HuggingFace client"""
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False
    }
    
    if seed is not None:
        params["seed"] = seed

    return await client.chat.completions.create(**params)

def format_response(response: Any) -> Dict[str, Any]:
    """Format HuggingFace response to match expected structure"""
    if hasattr(response, 'choices'):
        return {
            "choices": [{
                "message": {
                    "content": choice.message.content
                }
            } for choice in response.choices]
        }
    else:
        return {
            "choices": [{
                "message": {
                    "content": str(response)
                }
            }]
        }