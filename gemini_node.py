import sys
import os

# Add site-packages directory to Python's sys.path
'''
site_packages_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)
'''
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import logging
import random
import base64

from .env_utils import get_api_key
from .utils import ChatHistory
from .image_utils import (
    create_placeholder_image,
    prepare_batch_images,
    process_images_for_comfy,
)
from .response_utils import prepare_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiNode:
    def __init__(self):
        self.api_key = ""
        self.chat_history = ChatHistory()

        # Try to load API key from environment with better logging
        # First check system environment variable directly
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if self.api_key:
            logger.info("Successfully loaded Gemini API key from system environment")
        else:
            # Fall back to checking .env files
            self.api_key = get_api_key("GEMINI_API_KEY", "Gemini")
            if self.api_key:
                logger.info("Successfully loaded Gemini API key from .env file")
            else:
                logger.warning("No Gemini API key found in any location. You'll need to provide it in the node.")
        
        # Check for Google Generative AI SDK
        self.genai_available = self._check_genai_availability()

    def _check_genai_availability(self):
        """Check if Google Generative AI SDK is available"""
        try:
            # Import just to check availability
            from google import genai

            return True
        except ImportError:
            logger.error(
                "Google Generative AI SDK not installed. Install with: pip install google-generativeai"
            )
            return False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Create a vivid word-picture representation of this image include elements that characterize the subject, costume, prop elemts, the action, the background, layout and composition elements present on the scene, be sure to mention the style and mood of the scene. Like it would a film director or director of photography"}),
                "operation_mode": (
                    ["analysis", "generate_text", "generate_images"],
                    {"default": "generate_images"},
                ),
                "model_version": (
                    [
                        "gemini-2.0-flash-exp",
                        "gemini-2.0-pro",
                        "gemini-2.0-flash",
                        "gemini-2.0-flash-exp-image-generation",
                    ],
                    {"default": "gemini-2.0-flash-exp"},
                ),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "batch_count": ("INT", {"default": 4, "min": 1, "max": 8}),
                "aspect_ratio": (
                    ["none", "1:1", "16:9", "9:16", "4:3", "3:4", "5:4", "4:5"],
                    {"default": "none"},
                ),
                "external_api_key": ("STRING", {"default": ""}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "use_random_seed": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "generate_content"
    CATEGORY = "ImpactFrames💥🎞️/LLM"

    def generate_content(
        self,
        prompt,
        operation_mode="analysis",
        chat_mode=False,
        clear_history=False,
        images=None,
        video=None,
        audio=None,
        external_api_key="",
        max_images=6,
        batch_count=1,
        seed=0,
        max_output_tokens=8192,
        temperature=0.4,
        structured_output=False,
        aspect_ratio="none",
        use_random_seed=False,
        model_version="gemini-2.0-flash-exp",
    ):
        """Generate content using Gemini model with various input types."""

        # Check if Google Generative AI SDK is available
        if not self.genai_available:
            return (
                "ERROR: Google Generative AI SDK not installed. Install with: pip install"
                " google-generativeai",
                create_placeholder_image(),
            )

        # Import here to avoid ImportError during ComfyUI startup
        try:
            from google import genai
            from google.genai import types
            logger.info(f"Google Generative AI SDK path: {genai.__file__}")
        except ImportError:
            return ("ERROR: Failed to import Google Generative AI SDK", create_placeholder_image())

        # Use external API key if provided, otherwise use environment
        api_key = None
        if external_api_key and external_api_key.strip():
            api_key = external_api_key.strip()
            logger.info("Using API key provided in the node")
        else:
            # Directly check system environment variable first
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                logger.info("Using API key from system environment variable")
            elif self.api_key:
                api_key = self.api_key
                logger.info("Using API key from previously loaded environment")
            else:
                # Last attempt to load API key from .env file
                api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                if api_key:
                    self.api_key = api_key
                    logger.info("Successfully loaded Gemini API key from .env file")
                else:
                    logger.error("No API key available from any source")

        if not api_key:
            return (
                "ERROR: No API key provided. Please set GEMINI_API_KEY in your environment or"
                " provide it in the external_api_key field.",
                create_placeholder_image(),
            )

        if clear_history:
            self.chat_history.clear()

        # Use random seed if seed is 0 or use_random_seed is True
        if seed == 0 or use_random_seed:
            seed = random.randint(1, 2**31 - 1)
            logger.info(f"Using random seed: {seed}")
        else:
            # Ensure seed is within INT32 range (maximum value 2147483647)
            max_int32 = 2**31 - 1
            seed = seed % max_int32
            logger.info(f"Adjusted seed to INT32 range: {seed}")

        # Handle image generation mode
        if operation_mode == "generate_images":
            return self.generate_images(
                prompt=prompt,
                model_version=model_version
                if "image-generation" in model_version
                else "gemini-2.0-flash-exp-image-generation",
                images=images,
                batch_count=batch_count,
                temperature=temperature,
                seed=seed,
                max_images=max_images,
                aspect_ratio=aspect_ratio,
                use_random_seed=use_random_seed,
            )

        # Initialize the API client with the API key instead of using configure
        client = genai.Client(api_key=api_key)

        # Configure safety settings and generation parameters
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens, 
            temperature=temperature, 
            seed=seed,  # Add seed
            safety_settings=safety_settings  # Include safety settings in config
        )

        try:
            if chat_mode:
                # Handle chat mode with proper history
                history = self.chat_history.get_messages_for_api()

                # Create chat session
                chat_session = client.chats.create(
                    model=model_version,
                    history=history
                )

                # Create appropriate content parts based on input type
                contents = prepare_response(
                    prompt,
                    "image" if images is not None else "text",
                    None,
                    images,
                    video,
                    audio,
                    max_images,
                )
                # Extract content for chat format
                if (
                    isinstance(contents, list)
                    and len(contents) == 1
                    and isinstance(contents[0], dict)
                    and "parts" in contents[0]
                ):
                    contents = contents[0]["parts"]

                # Send message to chat and get response
                response = chat_session.send_message(
                    content=contents,
                    config=generation_config,
                )

                # Add to history and format response
                self.chat_history.add_message("user", prompt)
                self.chat_history.add_message("assistant", response.text)

                # Return the chat history
                generated_content = self.chat_history.get_formatted_history()

            else:
                # Standard non-chat mode - prepare content for each input type
                contents = prepare_response(
                    prompt,
                    "image" if images is not None else "text",
                    None,
                    images,
                    video,
                    audio,
                    max_images,
                )

                # Add structured output instruction if requested
                if structured_output:
                    if (
                        isinstance(contents, list)
                        and len(contents) > 0
                        and isinstance(contents[0], dict)
                        and "parts" in contents[0]
                        and len(contents[0]["parts"]) > 0
                    ):
                        if (
                            isinstance(contents[0]["parts"][0], dict)
                            and "text" in contents[0]["parts"][0]
                        ):
                            contents[0]["parts"][0][
                                "text"
                            ] = f"Please provide the response in a structured format. {contents[0]['parts'][0]['text']}"

                # Generate content using the model
                response = client.models.generate_content(
                    model=model_version,
                    contents=contents,
                    config=generation_config,
                )

                generated_content = response.text

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}", exc_info=True)
            generated_content = f"Error: {str(e)}"

        # For analysis mode, return the text response and an empty placeholder image
        return (generated_content, create_placeholder_image())

    def generate_images(
        self,
        prompt,
        model_version,
        images=None,
        batch_count=1,
        temperature=0.4,
        seed=0,
        max_images=6,
        aspect_ratio="none",
        use_random_seed=False,
    ):
        """Generate images using Gemini models with image generation capabilities"""
        try:
            # Import here to avoid ImportError during ComfyUI startup
            from google import genai
            from google.genai import types

            # Ensure we're using an image generation capable model
            if "image-generation" not in model_version:
                model_version = "gemini-2.0-flash-exp-image-generation"
                logger.info(f"Changed to image generation model: {model_version}")

            # Get API key - use the same logic as in generate_content
            api_key = None
            # Directly check system environment variable first
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                logger.info("Using API key from system environment variable for image generation")
            elif self.api_key:
                api_key = self.api_key
                logger.info("Using API key from previously loaded environment for image generation")
            else:
                # Last attempt to load API key from .env file
                api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                if api_key:
                    self.api_key = api_key
                    logger.info("Successfully loaded Gemini API key from .env file for image generation")
                else:
                    logger.error("No API key available from any source for image generation")
            
            if not api_key:
                return (
                    "ERROR: No API key available for image generation. Please set GEMINI_API_KEY in your environment or provide it in the external_api_key field.",
                    create_placeholder_image(),
                )

            # Create Gemini client
            client = genai.Client(api_key=api_key)

            # Generate a random seed if seed is 0 or use_random_seed is True
            if seed == 0 or use_random_seed:
                seed = random.randint(1, 2**31 - 1)
                logger.info(f"Using random seed for image generation: {seed}")
            else:
                # Ensure seed is within INT32 range (maximum value 2147483647)
                max_int32 = 2**31 - 1
                seed = seed % max_int32
                logger.info(f"Adjusted seed to INT32 range: {seed}")

            # Define aspect ratio dimensions for Imagen 3
            aspect_ratio_dimensions = {
                "none": (1024, 1024),  # Default square format
                "1:1": (1024, 1024),  # Square
                "16:9": (1408, 768),  # Landscape widescreen
                "9:16": (768, 1408),  # Portrait widescreen
                "4:3": (1280, 896),  # Standard landscape
                "3:4": (896, 1280),  # Standard portrait
                "5:4": (1024, 819),  # Medium landscape format
                "4:5": (819, 1024),  # Medium portrait format
            }

            # Get target dimensions based on aspect ratio
            target_width, target_height = aspect_ratio_dimensions.get(aspect_ratio, (1024, 1024))
            logger.info(f"Using resolution {target_width}x{target_height} for aspect ratio {aspect_ratio}")

            # Set up generation config with required fields
            gen_config_args = {
                "temperature": temperature,
                "response_modalities": ["Text", "Image"],  # Critical for image generation
                "seed": seed,  # Always include seed in config
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            }

            generation_config = types.GenerateContentConfig(**gen_config_args)

            # Prepare content for the API
            content = None

            # Process reference images if provided
            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                # Convert tensor to list of PIL images - resize to match target dimensions
                pil_images = prepare_batch_images(images, max_images, max_size=max(target_width, target_height))

                if len(pil_images) > 0:
                    # Construct prompt with specific dimensions
                    aspect_string = (
                        f" with dimensions {target_width}x{target_height}"
                        if aspect_ratio != "none"
                        else ""
                    )
                    content_text = (
                        f"Generate a new image in the style of these reference images{aspect_string}: {prompt}"
                    )

                    # Combine text and images
                    content = [content_text] + pil_images
                else:
                    logger.warning("No valid images found in input tensor")

            # Use text-only prompt if no images or processing failed
            if content is None:
                # Include specific dimensions in the prompt
                if aspect_ratio != "none":
                    content_text = (
                        f"Generate a detailed, high-quality image with dimensions {target_width}x{target_height}"
                        f" of: {prompt}"
                    )
                else:
                    content_text = f"Generate a detailed, high-quality image of: {prompt}"
                content = content_text

            # Track generated images
            all_generated_images = []
            status_text = ""

            # Generate images - handle batch generation with unique seeds
            for i in range(batch_count):
                try:
                    # Update seed for each batch (keep within INT32 range)
                    max_int32 = 2**31 - 1
                    current_seed = (seed + i) % max_int32
                    batch_config = types.GenerateContentConfig(
                        temperature=temperature,
                        response_modalities=["Text", "Image"],
                        seed=current_seed,
                        safety_settings=[
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ],
                    )

                    # Log the seed being used
                    logger.info(f"Generating batch {i+1} with seed {current_seed}")

                    # Generate content
                    response = client.models.generate_content(
                        model=model_version, contents=content, config=batch_config
                    )

                    # Process response to extract generated images and text
                    batch_images = []
                    response_text = ""

                    if hasattr(response, "candidates") and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                                for part in candidate.content.parts:
                                    # Extract text
                                    if hasattr(part, "text") and part.text:
                                        response_text += part.text + "\n"

                                    # Extract image data
                                    if hasattr(part, "inline_data") and part.inline_data:
                                        try:
                                            image_binary = part.inline_data.data
                                            batch_images.append(image_binary)
                                        except Exception as img_error:
                                            logger.error(
                                                f"Error extracting image from response: {str(img_error)}"
                                            )

                    if batch_images:
                        all_generated_images.extend(batch_images)
                        status_text += (
                            f"Batch {i+1} (seed {current_seed}): Generated {len(batch_images)} images\n"
                        )
                    else:
                        status_text += f"Batch {i+1} (seed {current_seed}): No images found in response\n"

                except Exception as batch_error:
                    status_text += f"Batch {i+1} error: {str(batch_error)}\n"

            # Process generated images into tensors for ComfyUI
            if all_generated_images:
                # Create a data structure for process_images_for_comfy
                image_data = {
                    "data": [
                        {"b64_json": base64.b64encode(img).decode("utf-8")}
                        for img in all_generated_images
                    ]
                }

                # Use the utility function to convert images
                image_tensors, mask_tensors = process_images_for_comfy(
                    image_data, response_key="data", field_name="b64_json"
                )

                # Get the actual resolution of the first image for information
                if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() >= 3:
                    height, width = image_tensors.shape[1:3]
                    resolution_info = f"Resolution: {width}x{height}"
                else:
                    resolution_info = ""

                result_text = (
                    f"Successfully generated {len(all_generated_images)} images using"
                    f" {model_version}.\n"
                )
                result_text += f"Prompt: {prompt}\n"
                result_text += f"Starting seed: {seed}\n"
                if resolution_info:
                    result_text += f"{resolution_info}\n"

                return result_text, image_tensors

            # No images were generated successfully
            return (
                f"No images were generated with {model_version}. Details:\n{status_text}",
                create_placeholder_image(),
            )

        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, create_placeholder_image()
