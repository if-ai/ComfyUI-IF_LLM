#deepseek_api.py
import aiohttp
import json
import logging
from typing import List, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)

async def send_deepseek_request(
    base64_images: Optional[List[str]], # Kept for interface compatibility but won't be used
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict]] = None,
) -> Dict[str, Any]:
    """
    Send a request to the DeepSeek API and return a unified response format.
    Note: This API currently supports text-only interactions.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Prepare messages (text-only)
        deepseek_messages = prepare_deepseek_messages(system_message, user_message, messages)

        data = {
            "model": model,
            "messages": deepseek_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": False,
            "response_format": {"type": "text"}
        }

        if seed is not None:
            data["seed"] = seed
        if tools:
            data["tools"] = tools
            if tool_choice:
                data["tool_choice"] = tool_choice

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=data
            ) as response:
                response.raise_for_status()
                response_data = await response.json()

                if tools:
                    return response_data
                
                choices = response_data.get('choices', [])
                if choices:
                    choice = choices[0]
                    message = choice.get('message', {})
                    generated_text = message.get('content', '')
                    return {
                        "choices": [{
                            "message": {
                                "content": generated_text
                            }
                        }]
                    }
                else:
                    error_msg = "Error: No valid choices in the DeepSeek response."
                    logger.error(error_msg)
                    return {"choices": [{"message": {"content": error_msg}}]}

    except aiohttp.ClientResponseError as e:
        error_msg = f"HTTP error occurred: {e.status}, message='{e.message}'"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}
    except Exception as e:
        error_msg = f"Exception during DeepSeek API call: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

def prepare_deepseek_messages(
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prepare messages for the DeepSeek API format (text-only).
    """
    deepseek_messages = []
    
    # Add system message if provided
    if system_message:
        deepseek_messages.append({"role": "system", "content": system_message})
    
    # Add previous conversation messages
    for message in messages:
        # Only include text content
        if isinstance(message.get("content"), str):
            deepseek_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
    
    # Add current user message
    deepseek_messages.append({"role": "user", "content": user_message})
    
    return deepseek_messages