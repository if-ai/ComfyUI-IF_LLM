import os
import io
import re
import yaml
import json
import torch
import torchvision
import cv2
import base64
import logging
import datetime
import requests
import time
import numpy as np
from io import BytesIO
from aiohttp import web
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageSequence
from typing import Tuple, Optional, Dict, Union, List, Any
import node_helpers
from torchvision.transforms import functional as TF
import folder_paths


from typing import Union, List, Tuple

logger = logging.getLogger(__name__)

def process_frames(self, images_tensor, frame_sample_count, max_pixels=512*512):
    """
    Process image frames for the model by sampling and resizing.

    Args:
        images_tensor: Input tensor in shape [B,H,W,C]
        frame_sample_count: Number of frames to sample
        max_pixels: Max pixels for each frame

    Returns:
        List of processed PIL images
    """
    try:
        # Get the batch size (number of frames)
        batch_size = images_tensor.shape[0]

        # Sample the frames evenly from the batch
        if batch_size <= frame_sample_count:
            # Use all frames if we have fewer than requested
            sampled_indices = list(range(batch_size))
        else:
            # Sample evenly across the frames
            sampled_indices = [
                int(i * (batch_size - 1) / (frame_sample_count - 1))
                for i in range(frame_sample_count)
            ]

        # Extract the sampled frames from the tensor
        sampled_frames = [images_tensor[i] for i in sampled_indices]

        # Convert to PIL images with proper preprocessing
        pil_images = []
        for frame in sampled_frames:
            # Convert tensor to numpy
            frame_np = frame.cpu().numpy()

            # Scale from [0,1] to [0,255] if needed
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

            # Convert to PIL
            pil_image = Image.fromarray(frame_np)

            # Ensure RGB mode
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Calculate resize dimensions if needed
            if max_pixels > 0:
                width, height = pil_image.size
                if width * height > max_pixels:
                    # Calculate new dimensions while maintaining aspect ratio
                    ratio = math.sqrt(max_pixels / (width * height))
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)

                    # Ensure dimensions are multiples of 32 for better compatibility
                    new_width = (new_width // 32) * 32
                    new_height = (new_height // 32) * 32

                    # Resize using LANCZOS for better quality
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

            # Add final checks
            if pil_image.size[0] > 2048 or pil_image.size[1] > 2048:
                # Limit maximum dimension to 2048
                pil_image.thumbnail((2048, 2048), Image.LANCZOS)

            pil_images.append(pil_image)

        return pil_images

    except Exception as e:
        logger.error(f"Error processing frames: {e}")
        # Return a single black frame as fallback
        fallback_size = (512, 512)
        return [Image.new("RGB", fallback_size, (0, 0, 0))]

def resize_image_max_side(img, max_size):
    """Resize image so its longest side is max_size while maintaining aspect ratio"""
    ratio = max_size / max(img.size)
    if ratio < 1:  # Only resize if image is larger than max_size
        new_size = tuple(int(dim * ratio) for dim in img.size)
        return img.resize(new_size, Image.LANCZOS)
    return img

def prepare_batch_images(images):
    """
    Convert images to list of batches.
    Handles tensor, list, and single image inputs while preserving dimensions.
    
    Args:
        images: torch.Tensor or list of tensors
        
    Returns:
        List of image tensors
    """
    try:
        if images is None:
            return []
            
        if isinstance(images, torch.Tensor):
            # Handle 4D tensor [B,H,W,C] - split into list of [H,W,C]
            if images.dim() == 4:
                return [images[i] for i in range(images.shape[0])]
            # Handle 3D tensor [H,W,C] - wrap in list
            elif images.dim() == 3:
                return [images]
            else:
                raise ValueError(f"Invalid tensor dimensions: {images.dim()}")
                
        # Handle list input - validate each element
        if isinstance(images, list):
            for i, img in enumerate(images):
                if not isinstance(img, torch.Tensor):
                    raise ValueError(f"Image {i} is not a tensor")
            return images
            
        # Handle single image
        return [images]
        
    except Exception as e:
        logger.error(f"Error in prepare_batch_images: {str(e)}")
        return []

def process_auto_mode_images(images, mask=None, batch_size=4):
    """
    Process images and masks for auto mode with proper mask dimensionality handling.
    
    Args:
        images: Input images tensor [B,H,W,C] or list of tensors
        mask: Mask tensor [B,H,W] or [B,1,H,W] or list of tensors
        batch_size: Maximum size of each batch (default 4)
        
    Returns:
        Tuple of (image_batches, mask_batches) where each is a list of tensors
    """
    try:
        # Convert images to list format
        if images is None or (isinstance(images, (list, tuple)) and len(images) == 0):
            # Return a tuple of empty lists
            return ([], [])

        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # [B,H,W,C]
                images = [images[i] for i in range(images.shape[0])]
            elif images.dim() == 3:  # [H,W,C]
                images = [images]
            else:
                raise ValueError(f"Invalid image tensor dimensions: {images.dim()}")
        
        # Split images into batches
        image_batches = []
        current_batch = []
        
        for img in images:
            if len(current_batch) == batch_size:
                image_batches.append(torch.stack(current_batch))
                current_batch = []
            current_batch.append(img)
            
        if current_batch:  # Don't forget the last batch
            image_batches.append(torch.stack(current_batch))

        # Process masks
        mask_batches = []
        
        if mask is not None:
            # Standardize mask format
            if isinstance(mask, torch.Tensor):
                # Handle different mask dimensions
                if mask.dim() == 2:  # [H,W]
                    mask = mask.unsqueeze(0)  # -> [1,H,W]
                elif mask.dim() == 3:  # [B,H,W] or [1,H,W]
                    if mask.shape[0] != len(images):
                        # Broadcast mask to match batch size
                        mask = mask.repeat(len(images), 1, 1)
                elif mask.dim() == 4:  # [B,1,H,W] or similar
                    mask = mask.squeeze(1)  # Remove channel dim -> [B,H,W]
                
                # Split mask into batches matching image batches
                start_idx = 0
                for img_batch in image_batches:
                    batch_size = img_batch.size(0)
                    mask_batch = mask[start_idx:start_idx + batch_size]
                    
                    mask_batches.append(mask_batch)
                    start_idx += batch_size
            else:
                # Handle list of masks
                mask_list = mask if isinstance(mask, list) else [mask] * len(images)
                start_idx = 0
                for img_batch in image_batches:
                    batch_size = img_batch.size(0)
                    mask_slice = mask_list[start_idx:start_idx + batch_size]
                    
                    # Convert and stack masks
                    mask_tensors = []
                    for m in mask_slice:
                        if isinstance(m, torch.Tensor):
                            if m.dim() == 2:
                                m = m.unsqueeze(0)  # Add batch dim
                            m = m.unsqueeze(-1)  # Add channel dim at end
                        else:
                            # Convert non-tensor masks
                            m = torch.tensor(m, dtype=torch.float32)
                            if m.dim() == 2:
                                m = m.unsqueeze(0).unsqueeze(-1)
                            elif m.dim() == 3:
                                m = m.unsqueeze(-1)
                        mask_tensors.append(m)
                    
                    mask_batch = torch.stack(mask_tensors)
                    mask_batches.append(mask_batch)
                    start_idx += batch_size
        else:
            # Create default masks matching image batches
            for img_batch in image_batches:
                mask_batch = torch.ones((img_batch.size(0), img_batch.size(1), 
                                       img_batch.size(2)),  # Removed extra dimension
                                  dtype=torch.float32,
                                  device=img_batch.device)
                mask_batches.append(mask_batch)

        return image_batches, mask_batches

    except Exception as e:
        logger.error(f"Error in process_auto_mode_images: {str(e)}")
        raise

def convert_images_for_api(images, target_format='tensor'):
    """
    Convert images to the specified format for API consumption.
    Supports conversion to: tensor, base64, pil
    """
    if images is None:
        return None

    # Handle single tensor input with ComfyUI compatibility
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:  # Single image
            images = images.unsqueeze(0)
        # Permute tensor to ComfyUI format (B, H, W, C) -> (B, C, H, W)
        images = images.permute(0, 3, 1, 2)

        if target_format == 'tensor':
            return images
        elif target_format == 'base64':
            return [tensor_to_base64(img) for img in images]
        elif target_format == 'pil':
            return [TF.to_pil_image(img) for img in images]
        else:
            raise ValueError(f"Unsupported target format for tensor: {target_format}")

    # Handle list of tensors input
    elif isinstance(images, list) and all(isinstance(x, torch.Tensor) for x in images):
        # Filter out tensors with unsupported channel counts
        supported_images = []
        for idx, img in enumerate(images):
            if img.shape[0] in [1, 3]:
                supported_images.append(img)
            elif img.shape[0] > 3:
                logger.warning(f"Skipping tensor at index {idx} with {img.shape[0]} channels.")
            else:
                logger.warning(f"Skipping tensor at index {idx} with unsupported number of channels: {img.shape[0]}")
        if not supported_images:
            raise ValueError("No supported image tensors found in the input list.")

        if target_format == 'tensor':
            return torch.stack(supported_images).permute(0, 3, 1, 2)  # Ensure correct format
        elif target_format == 'base64':
            return [tensor_to_base64(img) for img in supported_images]
        elif target_format == 'pil':
            return [TF.to_pil_image(img) for img in supported_images]
        else:
            raise ValueError(f"Unsupported target format for list of tensors: {target_format}")

    # Handle base64 input
    elif isinstance(images, str) or (isinstance(images, list) and all(isinstance(x, str) for x in images)):
        base64_list = [images] if isinstance(images, str) else images
        if target_format == 'base64':
            return base64_list

        # Convert base64 to PIL first
        pil_images = [base64_to_pil(b64) for b64 in base64_list]
        if target_format == 'pil':
            return pil_images
        elif target_format == 'tensor':
            tensors = [pil_to_tensor(img) for img in pil_images]
            return torch.stack(tensors).permute(0, 2, 3, 1)  # Convert to ComfyUI format (B,H,W,C)
        else:
            raise ValueError(f"Unsupported target format for base64 input: {target_format}")

    # Handle list of PIL images input
    elif isinstance(images, (list, tuple)) and all(isinstance(x, Image.Image) for x in images):
        if target_format == 'pil':
            return images
        elif target_format == 'base64':
            return [pil_image_to_base64(img) for img in images]
        elif target_format == 'tensor':
            tensors = [pil_to_tensor(img) for img in images]
            return torch.stack(tensors).permute(0, 2, 3, 1)  # Maintain ComfyUI format
        else:
            raise ValueError(f"Unsupported target format for PIL input: {target_format}")

    # If none of the above conditions are met, attempt to convert using the default method
    # Ensure that images can be saved (i.e., are PIL Images)
    else:
        try:
            encoded_images = []
            for img in images:
                if not isinstance(img, Image.Image):
                    raise ValueError(f"Expected PIL.Image, got {type(img)}")
                buffered = BytesIO()
                img.save(buffered, format="PNG")  # Adjust format if needed
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                encoded_images.append(img_str)
            return encoded_images
        except Exception as e:
            raise ValueError(f"Unsupported image format or target format: {target_format}. Error: {str(e)}") from e

def convert_single_image(image, target_format):
    """Helper function to convert a single image"""
    if isinstance(image, str) and image.startswith('data:image'):
        # Convert base64 to PIL
        base64_data = image.split('base64,')[1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
    
    if target_format == 'pil':
        return image
    elif target_format == 'tensor':
        return pil_to_tensor(image)
    elif target_format == 'base64':
        return pil_image_to_base64(image)

def load_placeholder_image(placeholder_image_path):
        
        # Ensure the placeholder image exists
        if not os.path.exists(placeholder_image_path):
            # Create a proper RGB placeholder image
            placeholder = Image.new('RGB', (512, 512), color=(73, 109, 137))
            os.makedirs(os.path.dirname(placeholder_image_path), exist_ok=True)
            placeholder.save(placeholder_image_path)
        
        img = node_helpers.pillow(Image.open, placeholder_image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

def process_images_for_comfy(images, placeholder_image_path=None, response_key='data', field_name='b64_json', field2_name=""):
    """Process images for ComfyUI, ensuring consistent sizes."""
    def _process_single_image(image):
        try:
            if image is None:
                return load_placeholder_image(placeholder_image_path)

            # Handle JSON/API response
            if isinstance(image, dict):
                try:
                    # Only attempt to extract from response if response_key is provided
                    if response_key and response_key in image:
                        items = image[response_key]
                        if isinstance(items, list):
                            for item in items:
                                # Only attempt to get field_name if it's provided
                                if field2_name and field_name:
                                    image_data = item.get(field2_name, {}).get(field_name)
                                elif field_name:
                                    image_data = item.get(field_name)
                                else:
                                    continue
                                
                                if image_data:
                                    # Convert the first valid image found
                                    if isinstance(image_data, str):
                                        if image_data.startswith(('data:image', 'http:', 'https:')):
                                            image = image_data  # Will be handled by URL processing below
                                        else:
                                            # Handle base64 directly
                                            image_data = base64.b64decode(image_data)
                                            image = Image.open(BytesIO(image_data))
                                            break
                    
                    if isinstance(image, dict):
                        logger.warning(f"No valid image found in response under key '{response_key}'")
                        return load_placeholder_image(placeholder_image_path)
                except Exception as e:
                    logger.error(f"Error processing API response: {str(e)}")
                    return load_placeholder_image(placeholder_image_path)

            # Convert various input types to PIL Image
            if isinstance(image, torch.Tensor):
                # Ensure tensor is in correct format [B,H,W,C] or [H,W,C]
                if image.dim() == 4:
                    if image.shape[-1] != 3:  # Wrong channel dimension
                        image = image.squeeze(1)  # Remove channel dim if [B,1,H,W]
                        if image.shape[-1] != 3:  # Still wrong shape
                            image = image.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
                    image = image.squeeze(0)  # Remove batch dim
                elif image.dim() == 3 and image.shape[0] == 3:
                    image = image.permute(1, 2, 0)  # [C,H,W] -> [H,W,C]
                
                # Convert to numpy and scale to 0-255 range
                image = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(image)
                
            elif isinstance(image, np.ndarray):
                # Handle numpy arrays
                if image.dtype != np.uint8:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                if image.shape[-1] != 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(image)
                
            elif isinstance(image, str):
                if image.startswith('data:image'):
                    base64_data = image.split('base64,')[1]
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                elif image.startswith(('http:', 'https:')):
                    response = requests.get(image)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    image = Image.open(image).convert('RGB')

            # Ensure we have a PIL Image at this point
            if not isinstance(image, Image.Image):
                raise ValueError(f"Failed to convert to PIL Image: {type(image)}")

            # Convert PIL to tensor in ComfyUI format
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)
            
            # Ensure NHWC format
            if img_tensor.dim() == 3:  # [H,W,C]
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dim: [1,H,W,C]
            
            # Create mask
            mask_tensor = torch.ones((1, img_tensor.shape[1], img_tensor.shape[2]), 
                                   dtype=torch.float32)

            return img_tensor, mask_tensor

        except Exception as e:
            logger.error(f"Error processing single image: {str(e)}")
            return load_placeholder_image(placeholder_image_path)

    try:
        # Handle API responses
        if isinstance(images, dict) and response_key in images:
            # Process each item in API response
            all_tensors = []
            all_masks = []
            
            items = images[response_key]
            if isinstance(items, list):
                for item in items:
                    try:
                        img_tensor, mask_tensor = _process_single_image({response_key: [item]})
                        all_tensors.append(img_tensor)
                        all_masks.append(mask_tensor)
                    except Exception as e:
                        logger.error(f"Error processing response item: {str(e)}")
                        continue
                
                if all_tensors:
                    return torch.cat(all_tensors, dim=0), torch.cat(all_masks, dim=0)
            
            # If no valid images processed, return placeholder
            return load_placeholder_image(placeholder_image_path)

        # Handle list/batch of images
        if isinstance(images, (list, tuple)):
            all_tensors = []
            all_masks = []
            
            for img in images:
                try:
                    img_tensor, mask_tensor = _process_single_image(img)
                    all_tensors.append(img_tensor)
                    all_masks.append(mask_tensor)
                except Exception as e:
                    logger.error(f"Error processing batch image: {str(e)}")
                    continue
            
            if all_tensors:
                return torch.cat(all_tensors, dim=0), torch.cat(all_masks, dim=0)
            
            return load_placeholder_image(placeholder_image_path)

        # Handle single image
        return _process_single_image(images)

    except Exception as e:
        logger.error(f"Error in process_images_for_comfy: {str(e)}")
        return _process_single_image(None)

def process_mask(retrieved_mask, image_tensor):
    """
    Process the retrieved_mask to ensure it's in the correct format.
    The mask should be a tensor of shape (B, H, W), matching image_tensor's batch size and dimensions.
    """
    try:
        # Handle torch.Tensor
        if isinstance(retrieved_mask, torch.Tensor):
            # Normalize dimensions
            if retrieved_mask.dim() == 2:  # (H, W)
                retrieved_mask = retrieved_mask.unsqueeze(0)  # Add batch dimension
            elif retrieved_mask.dim() == 3:
                if retrieved_mask.shape[0] != image_tensor.shape[0]:
                    # Adjust batch size
                    retrieved_mask = retrieved_mask.repeat(image_tensor.shape[0], 1, 1)
            elif retrieved_mask.dim() == 4:
                # If mask has a channel dimension, reduce it
                retrieved_mask = retrieved_mask.squeeze(1)
            else:
                raise ValueError(f"Invalid mask tensor dimensions: {retrieved_mask.shape}")

            # Ensure proper format
            retrieved_mask = retrieved_mask.float()
            if retrieved_mask.max() > 1.0:
                retrieved_mask = retrieved_mask / 255.0

            # Ensure mask dimensions match image dimensions
            if retrieved_mask.shape[1:] != image_tensor.shape[2:]:
                # Resize mask to match image dimensions
                retrieved_mask = torch.nn.functional.interpolate(
                    retrieved_mask.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return retrieved_mask

        # Handle PIL Image
        elif isinstance(retrieved_mask, Image.Image):
            mask_array = np.array(retrieved_mask.convert('L')).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

            # Adjust batch size
            if mask_tensor.shape[0] != image_tensor.shape[0]:
                mask_tensor = mask_tensor.repeat(image_tensor.shape[0], 1, 1)

            # Resize if needed
            if mask_tensor.shape[1:] != image_tensor.shape[2:]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return mask_tensor

        # Handle numpy array
        elif isinstance(retrieved_mask, np.ndarray):
            mask_array = retrieved_mask.astype(np.float32)
            if mask_array.max() > 1.0:
                mask_array = mask_array / 255.0
            if mask_array.ndim == 2:
                pass  # (H, W)
            elif mask_array.ndim == 3:
                mask_array = np.mean(mask_array, axis=2)  # Convert to grayscale
            else:
                raise ValueError(f"Invalid mask array dimensions: {mask_array.shape}")

            mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

            # Adjust batch size
            if mask_tensor.shape[0] != image_tensor.shape[0]:
                mask_tensor = mask_tensor.repeat(image_tensor.shape[0], 1, 1)

            # Resize if needed
            if mask_tensor.shape[1:] != image_tensor.shape[2:]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return mask_tensor

        # Handle other types (e.g., file paths, base64 strings)
        elif isinstance(retrieved_mask, str):
            # Attempt to process as file path or base64 string
            if os.path.exists(retrieved_mask):
                pil_image = Image.open(retrieved_mask).convert('L')
            elif retrieved_mask.startswith('data:image'):
                base64_data = retrieved_mask.split('base64,')[1]
                image_data = base64.b64decode(base64_data)
                pil_image = Image.open(BytesIO(image_data)).convert('L')
            else:
                raise ValueError(f"Invalid mask string: {retrieved_mask}")
            return process_mask(pil_image, image_tensor)

        else:
            raise ValueError(f"Unsupported mask type: {type(retrieved_mask)}")

    except Exception as e:
        logger.error(f"Error processing mask: {str(e)}")
        # Return a default mask matching the image dimensions
        return torch.ones((image_tensor.shape[0], image_tensor.shape[2], image_tensor.shape[3]), dtype=torch.float32)

def convert_mask_to_grayscale_alpha(mask_input):
    """
    Convert mask to grayscale alpha channel.
    Handles tensors, PIL images and numpy arrays.
    Returns tensor in shape [B,1,H,W].
    """
    if isinstance(mask_input, torch.Tensor):
        # Handle tensor input
        if mask_input.dim() == 2:  # [H,W]
            return mask_input.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif mask_input.dim() == 3:  # [C,H,W] or [B,H,W]
            if mask_input.shape[0] in [1,3,4]:  # Assume channel-first
                if mask_input.shape[0] == 4:  # Use alpha channel
                    return mask_input[3:4].unsqueeze(0)
                else:  # Convert to grayscale
                    weights = torch.tensor([0.299, 0.587, 0.114]).to(mask_input.device)
                    return (mask_input * weights.view(-1,1,1)).sum(0).unsqueeze(0).unsqueeze(0)
        elif mask_input.dim() == 4:  # [B,C,H,W]
            if mask_input.shape[1] == 4:  # Use alpha channel
                return mask_input[:,3:4]
            else:  # Convert to grayscale
                weights = torch.tensor([0.299, 0.587, 0.114]).to(mask_input.device)
                return (mask_input * weights.view(1,-1,1,1)).sum(1).unsqueeze(1)
                
    elif isinstance(mask_input, Image.Image):
        # Convert PIL image to grayscale
        mask = mask_input.convert('L')
        tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
    elif isinstance(mask_input, np.ndarray):
        # Handle numpy array
        if mask_input.ndim == 2:  # [H,W]
            tensor = torch.from_numpy(mask_input).float()
            return tensor.unsqueeze(0).unsqueeze(0)
        elif mask_input.ndim == 3:  # [H,W,C]
            if mask_input.shape[2] == 4:  # Use alpha channel
                tensor = torch.from_numpy(mask_input[:,:,3]).float()
            else:  # Convert to grayscale
                tensor = torch.from_numpy(np.dot(mask_input[...,:3], [0.299, 0.587, 0.114])).float()
            return tensor.unsqueeze(0).unsqueeze(0)
            
    raise ValueError(f"Unsupported mask input type: {type(mask_input)}")

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a tensor to a base64-encoded PNG image string."""
    try:
        # Ensure the tensor is in [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)

        # Handle different tensor dimensions
        if tensor.dim() == 3:
            # [C, H, W]
            if tensor.shape[0] == 1:
                # Grayscale image, convert to RGB by repeating channels
                image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                image = np.repeat(image, 3, axis=2)
            elif tensor.shape[0] == 3:
                # RGB image
                image = tensor.permute(1, 2, 0).cpu().numpy()
            else:
                # Handle tensors with more than 3 channels: select the first 3 channels
                logger.warning(f"Unsupported number of channels: {tensor.shape[0]}. Selecting first 3 channels.")
                if tensor.shape[0] >= 3:
                    image = tensor[:3, :, :].permute(1, 2, 0).cpu().numpy()
                else:
                    raise ValueError(f"Unsupported number of channels: {tensor.shape[0]}")
        elif tensor.dim() == 2:
            # [H, W] Grayscale image
            image = tensor.unsqueeze(-1).cpu().numpy()
            image = np.repeat(image, 3, axis=2)
        else:
            raise ValueError(f"Unsupported tensor shape for conversion: {tensor.shape}")

        # Convert to uint8
        image = (image * 255).astype(np.uint8)

        # Create PIL Image
        pil_image = Image.fromarray(image)

        # Save image to buffer
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        logger.error(f"Error converting tensor to base64: {str(e)}", exc_info=True)
        raise

def tensor_to_pil(tensor):
    """
    Convert a tensor to a PIL image with better error handling and format detection.
    
    Args:
        tensor: A PyTorch tensor representing an image
        
    Returns:
        PIL.Image: The converted PIL image
    """
    try:
        # Ensure tensor is on CPU
        tensor = tensor.cpu()
        
        # Handle different tensor shapes
        if tensor.dim() == 4 and tensor.shape[0] == 1:  # [1, C, H, W] or [1, H, W, C]
            tensor = tensor.squeeze(0)  # Remove batch dimension
            
        # Determine if we have a channels-first or channels-last format
        if tensor.dim() == 3:
            # Handle both [C, H, W] and [H, W, C] formats
            if tensor.shape[0] in [1, 3, 4]:  # Channels-first format [C, H, W]
                tensor = tensor.permute(1, 2, 0)  # Convert to [H, W, C]
                
        # Special case for grayscale
        if tensor.dim() == 2:
            # Add a channel dimension for grayscale [H, W] -> [H, W, 1]
            tensor = tensor.unsqueeze(-1)
            
        # Convert to numpy array
        tensor_np = tensor.numpy()
        
        # Scale to 0-255 range for uint8
        tensor_np = np.clip(tensor_np * 255, 0, 255).astype(np.uint8)
        
        # Create PIL image
        pil_image = Image.fromarray(tensor_np)
        return pil_image
        
    except Exception as e:
        logger.error(f"Error in tensor_to_pil: {e}")
        raise ValueError(f"Failed to convert tensor to PIL image: {e}")

def pil_to_tensor(pil_image):
    # Convert PIL image to tensor
    tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
    return tensor.permute(2, 0, 1) if tensor.dim() == 3 else tensor.unsqueeze(0)

def base64_to_pil(base64_str):
    """Convert base64 string to PIL Image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split('base64,')[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

def pil_image_to_base64(pil_image: Image.Image) -> str:
    """Converts a PIL Image to a data URL."""
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to data URL: {str(e)}", exc_info=True)
        raise

def clean_text(generated_text, remove_weights=True, remove_author=True):
    """Clean text while preserving intentional line breaks."""
    # Split into lines first to preserve breaks
    lines = generated_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if line.strip():  # Only process non-empty lines
            # Remove author attribution if requested
            if remove_author:
                line = re.sub(r"\bby:.*", "", line)

            # Remove weights if requested
            if remove_weights:
                line = re.sub(r"\(([^)]*):[\d\.]*\)", r"\1", line)
                line = re.sub(r"(\w+):[\d\.]*(?=[ ,]|$)", r"\1", line)

            # Remove markup tags
            line = re.sub(r"<[^>]*>", "", line)

            # Remove lonely symbols and formatting
            line = re.sub(r"(?<=\s):(?=\s)", "", line)
            line = re.sub(r"(?<=\s);(?=\s)", "", line)
            line = re.sub(r"(?<=\s),(?=\s)", "", line)
            line = re.sub(r"(?<=\s)#(?=\s)", "", line)

            # Clean up extra spaces while preserving line structure
            line = re.sub(r"\s{2,}", " ", line)
            line = re.sub(r"\.,", ",", line)
            line = re.sub(r",,", ",", line)

            # Remove audio tags from the line
            if "<audio" in line:
                print(f"iF_prompt_MKR: Audio has been generated.")
                line = re.sub(r"<audio.*?>.*?</audio>", "", line)

            cleaned_lines.append(line.strip())

    # Join with newlines to preserve line structure
    return "\n".join(cleaned_lines)

def get_api_key(api_key_name, engine):
    """
    Retrieve API key from environment variables or .env file.
    
    Args:
        api_key_name (str): Name of the API key environment variable
        engine (str): Name of the engine being used
        
    Returns:
        str: API key if found and valid
        
    Raises:
        ValueError: If API key is missing or invalid
    """
    local_engines = ["ollama", "llamacpp", "kobold", "lmstudio", "textgen", "sentence_transformers", "transformers"]
    # Try to get the key from .env first
    load_dotenv()
    api_key = os.getenv(api_key_name)
    if engine.lower() in local_engines:
        print(f"You are using {engine} as the engine, no API key is required.")
        return "1234"
    
    # Special handling for HuggingFace
    if engine.lower() == "huggingface":
        # Try both conventional and HF-specific env var names
        api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_AUTH_TOKEN")
        if api_key:
            if validate_huggingface_token(api_key):
                return api_key
            else:
                raise ValueError("Invalid HuggingFace API key")
        raise ValueError("No HuggingFace API key found in environment variables")
    
    elif api_key:
        print(f"API key for {api_key_name} found in .env file or environment variables")
        return api_key
    
    print(f"API key for {api_key_name} not found in .env file or environment variables")
    raise ValueError(f"{api_key_name} not found. Please set it in your .env file or as an environment variable.")

def get_models(engine, base_ip, port, api_key):

    if engine == "ollama":
        api_url = f"http://{base_ip}:{port}/api/tags"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            return models
        except Exception as e:
            print(f"Failed to fetch models from Ollama: {e}")
            return []
    
    
    elif engine == "huggingface":
        fallback_models = [
            # Vision Language Models (VLM)
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "Qwen/Qwen2-VL-7B-Chat",
            "Qwen/Qwen2-VL-7B",
            "Qwen/Qwen2-VL-2B-Chat",
            "Qwen/Qwen2-VL-2B",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
            "microsoft/phi-2",
            "HuggingFaceH4/zephyr-7b-beta",
            
            # Text to Image Models
            "stabilityai/sdxl-turbo",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-3-base",
            "stabilityai/stable-diffusion-3-medium",
            "stabilityai/stable-diffusion-3-small",
            "black-forest-labs/FLUX.1-dev",
            "playgroundai/playground-v2-256px",
            "playgroundai/playground-v2-1024px",
            
            # Image to Image Models
            "timbrooks/instruct-pix2pix",
            "lambdalabs/sd-image-variations-diffusers",
            "diffusers/controlnet-canny-sdxl-1.0",
            
            # Specialized Models
            "kandinsky-community/kandinsky-3",
            "stabilityai/stable-cascade",
            "dataautogpt3/OpenDalle3",
            "ByteDance/SDXL-Lightning",
            
            # ControlNet Models
            "lllyasviel/control_v11p_sd15_canny",
            "lllyasviel/control_v11p_sd15_openpose",
            "lllyasviel/control_v11p_sd15_depth",
            
            # Text Feature Extraction
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            
            # Image Feature Extraction  
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            
            # Text Classification
            "distilbert-base-uncased-finetuned-sst-2-english",
            "roberta-base-openai-detector",
            
            # Text Generation
            "gpt2",
            "facebook/opt-350m",
            
            # Translation 
            "Helsinki-NLP/opus-mt-en-fr",
            "Helsinki-NLP/opus-mt-fr-en",
            
            # Question Answering
            "deepset/roberta-base-squad2",
            "distilbert-base-cased-distilled-squad"
        ]

        try:
            # Verify API key
            if not api_key or api_key == "1234":
                print("No valid HuggingFace API key provided. Using fallback models.")
                return fallback_models

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }

            # Check inference API endpoint directly
            inference_url = "https://api-inference.huggingface.co/status"
            response = requests.get(inference_url, headers=headers)
            
            if response.status_code != 200:
                print("Failed to verify HuggingFace Inference API access. Using fallback models.")
                return fallback_models

            # Get models available for inference API
            models_url = "https://api-inference.huggingface.co/framework/all"
            response = requests.get(models_url, headers=headers)

            if response.status_code == 200:
                api_models = []
                data = response.json()
                
                # Extract models supporting inference API
                for framework in data:
                    for model in framework.get("models", []):
                        model_id = model.get("model_id")
                        if model_id:
                            api_models.append(model_id)
                
                # Combine with fallback models and remove duplicates
                combined_models = list(dict.fromkeys(api_models + fallback_models))
                return combined_models
            else:
                print(f"Failed to fetch inference models. Status code: {response.status_code}")
                return fallback_models

        except Exception as e:
            print(f"Error fetching HuggingFace models: {str(e)}")
            return fallback_models

    elif engine == "deepseek":
        fallback_models = [
            "deepseek-reasoner",
            "deepseek-chat",
            "deepseek-coder"
        ]

        #api_key = get_api_key("DEEPSEEK_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid DeepSeek API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.deepseek.com/v1/models"  # Adjust URL if needed
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from DeepSeek API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from DeepSeek: {e}")
            print(f"Returning fallback list of {len(fallback_models)} DeepSeek models")
            return fallback_models

    elif engine == "lmstudio":
        api_url = f"http://{base_ip}:{port}/v1/models"
        try:
            print(f"Attempting to connect to {api_url}")
            response = requests.get(api_url, timeout=10)
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data["data"]]
                return models
            else:
                print(f"Failed to fetch models from LM Studio. Status code: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to LM Studio server: {e}")
            return []

    elif engine == "textgen":
        api_url = f"http://{base_ip}:{port}/v1/internal/model/list"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = response.json()["model_names"]
            return models
        except Exception as e:
            print(f"Failed to fetch models from text-generation-webui: {e}")
            return []

    elif engine == "kobold":
        api_url = f"http://{base_ip}:{port}/api/v1/model"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            model = response.json()["result"]
            return [model]
        except Exception as e:
            print(f"Failed to fetch models from Kobold: {e}")
            return []

    elif engine == "llamacpp":
        api_url = f"http://{base_ip}:{port}/v1/models"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = [model["id"] for model in response.json()["data"]]
            return models
        except Exception as e:
            print(f"Failed to fetch models from llama.cpp: {e}")
            return []

    elif engine == "vllm":
        api_url = f"http://{base_ip}:{port}/v1/models"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            # Adapt this based on vLLM"s actual API response structure
            models = [model["id"] for model in response.json()["data"]] 
            return models
        except Exception as e:
            print(f"Failed to fetch models from vLLM: {e}")
            return []

    elif engine == "openai":
        fallback_models = [
            # GPT-4o Models
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-audio-preview",
            "gpt-4o-audio-preview-2024-10-01",
            "gpt-4o-audio-preview-2024-12-17",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini-audio-preview",
            "gpt-4o-mini-audio-preview-2024-12-17",
            "gpt-4o-mini-realtime-preview",
            "gpt-4o-mini-realtime-preview-2024-12-17",
            "gpt-4o-realtime-preview",
            "gpt-4o-realtime-preview-2024-10-01",
            "gpt-4o-realtime-preview-2024-12-17",

            # GPT-4 Models
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",

            # GPT-3.5 Models
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",

            # DALL-E Models
            "dall-e-2",
            "dall-e-3",

            # Whisper Models
            "whisper-1",
            "whisper-I",

            # TTS Models
            "tts-1",
            "tts-1-1106",
            "tts-1-hd",
            "tts-1-hd-1106",
            "tts-l-hd",

            # Embedding Models
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",

            # Specialized Models
            "babbage-002",
            "chatgpt-4o-latest",
            "davinci-002",
            "gpt40-0806-loco-vm",

            # O1 Models
            "o1",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o1-preview",
            "o1-preview-2024-09-12",

            # Omni Moderation
            "omni-moderation-2024-09-26",
            "omni-moderation-latest",

            # Future/Experimental
            "gpt-4.5-preview",
            "gpt-4.5-preview-2025-02-27",
            "o3-mini",
            "o3-mini-2025-01-31"
        ]

        #api_key = get_api_key("OPENAI_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid OpenAI API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.openai.com/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from OpenAI API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from OpenAI: {e}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response"):
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            print(f"Returning fallback list of {len(fallback_models)} OpenAI models")
            return fallback_models
    
    elif engine == "xai":
        fallback_models = [
            "grok-2",
            "grok-2-1212",
            "grok-2-latest",
            "grok-2-vision",
            "grok-2-vision-1212",
            "grok-2-vision-latest",
            "grok-beta",
            "grok-vision-beta"
        ]

        #api_key = get_api_key("XAI_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid XAI API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.x.ai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from XAI API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from XAI: {e}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response"):
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            print(f"Returning fallback list of {len(fallback_models)} XAI models")
            return fallback_models

    elif engine == "mistral":
        fallback_models = [
            "codestral-2405",
            "codestral-2411-rc5",
            "codestral-2412",
            "codestral-2501",
            "codestral-latest",
            "codestral-mamba-2407",
            "codestral-mamba-latest",
            "ministral-3b-2410",
            "ministral-3b-latest",
            "ministral-8b-2410",
            "ministral-8b-latest",
            "mistral-embed",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-large-2411",
            "mistral-large-latest",
            "mistral-large-pixtral-2411",
            "mistral-medium",
            "mistral-medium-2312",
            "mistral-medium-latest",
            "mistral-moderation-2411",
            "mistral-moderation-latest",
            "mistral-ocr-2503",
            "mistral-ocr-latest",
            "mistral-saba-2502",
            "mistral-saba-latest",
            "mistral-small",
            "mistral-small-2312",
            "mistral-small-2402",
            "mistral-small-2409",
            "mistral-small-2501",
            "mistral-small-latest",
            "mistral-tiny",
            "mistral-tiny-2312",
            "mistral-tiny-2407",
            "mistral-tiny-latest",
            "open-codestral-mamba",
            "open-mistral-7b",
            "open-mistral-nemo",
            "open-mistral-nemo-2407",
            "open-mixtral-8x22b",
            "open-mixtral-8x22b-2404",
            "open-mixtral-8x7b",
            "pixtral-12b",
            "pixtral-12b-2409",
            "pixtral-12b-latest",
            "pixtral-large-2411",
            "pixtral-large-latest"
        ]

        #api_key = get_api_key("MISTRAL_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid Mistral API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.mistral.ai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from Mistral API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from Mistral: {e}")
            print(f"Returning fallback list of {len(fallback_models)} Mistral models")
            return fallback_models

    elif engine == "groq":
        fallback_models = [
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            "distil-whisper-large-v3-en",
            "gemma2-9b-it",
            "llama-guard-3-8b",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "llama-3.3-70b-specdec",
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "llava-v1.5-7b-4096-preview",
            "mixtral-8x7b-32768",
            "mistral-saba-24b",
            "qwen-2.5-32b",
            "qwen-2.5-coder-32b",
            "qwen-qwq-32b",
            "whisper-large-v3",
            "whisper-large-v3-turbo"
        ]

        #api_key = get_api_key("GROQ_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid GROQ API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.groq.com/openai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from GROQ API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from GROQ: {e}")
            print(f"Returning fallback list of {len(fallback_models)} GROQ models")
            return fallback_models

    elif engine == "anthropic":
        return [
            "claude-3-5-opus-latest",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20240620",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-latest",
            "claude-3-5-haiku-20241022"
        ]

    elif engine == "gemini":
        return [
            "learnlrn-1.5-pro-experimental",
            "gemini-2.0-flash-thinking-exp-1219",
            "gemini-2.0-flash-exp",
            "gemini-exp-1206",
            "gemini-exp-1121",
            "gemini-exp-1114",
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b-exp-0924",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-latest",
            "gemini-pro",
            "gemini-pro-vision",
        ]

    elif engine == "sentence_transformers":
        return [
            "sentence-transformers/all-MiniLM-L6-v2",
            "avsolatorio/GIST-small-Embedding-v0",
        ]

    elif engine == "transformers":
        # Standard list of transformers models to show
        fallback_models = [
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",  # Default model we want to use
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "Qwen/QwQ-32B-AWQ",  # Keep QwQ-32B-AWQ model
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-72B-Instruct"
        ]
        
        # Check if we have a transformers model manager to list models
        try:
            from transformers_api import _transformers_manager
            
            # Get list of models from LLM directory
            try:
                # Get models directory dynamically
                try:
                    import folder_paths
                    models_dir = folder_paths.models_dir
                except (ImportError, AttributeError):
                    # Fallback to a default location if folder_paths is not available
                    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                    os.makedirs(models_dir, exist_ok=True)
                    print(f"Could not import folder_paths.models_dir, using fallback: {models_dir}")
                
                llm_path = os.path.join(models_dir, "LLM")
                if os.path.exists(llm_path) and os.path.isdir(llm_path):
                    # List directories in the LLM folder
                    local_models = []
                    for model_dir in os.listdir(llm_path):
                        model_path = os.path.join(llm_path, model_dir)
                        if os.path.isdir(model_path):
                            # Check if it has config.json to verify it's a model
                            if os.path.exists(os.path.join(model_path, "config.json")):
                                if "/" not in model_dir and "\\" not in model_dir:
                                    # For non-namespaced models, use the directory name
                                    local_models.append(model_dir)
                                else:
                                    # For models with namespaces, keep the structure
                                    local_models.append(model_dir)
                    
                    # If we found local models, add them to our list
                    if local_models:
                        print(f"Found {len(local_models)} local transformers models")
                        # Combine local models with fallback models (local models first)
                        combined_models = list(dict.fromkeys(local_models + fallback_models))
                        return combined_models
            except Exception as e:
                print(f"Error scanning local models directory: {e}")
            
            # If we couldn't find local models, return fallback list
            return fallback_models
        except ImportError:
            print("TransformersModelManager not available, using fallback models list")
            return fallback_models

    else:
        print(f"Unsupported engine - {engine}")
        return []

def validate_models(model, provider, model_type, base_ip, port, api_key):
        available_models = get_models(provider, base_ip, port, api_key)
        if available_models is None or model not in available_models:
            error_message = f"Invalid {model_type} model selected: {model} for provider {provider}. Available models: {available_models}"
            print(error_message)
            raise ValueError(error_message)

class EnhancedYAMLDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(EnhancedYAMLDumper, self).increase_indent(flow, False)

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

EnhancedYAMLDumper.add_representer(str, str_presenter)


def validate_huggingface_token(api_key):
    """Validate HuggingFace API token"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        # Try to access the API with the token
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error validating HuggingFace token: {e}")
        return False

def get_huggingface_url(model_or_url):
    """Convert model name to full HuggingFace API URL if needed"""
    if model_or_url.startswith(('http://', 'https://')):
        return model_or_url
    return f'https://api-inference.huggingface.co/models/{model_or_url}'

def send_huggingface_request(endpoint, payload, api_key, max_retries=3):
    """Send request to HuggingFace Inference API with retry logic"""
    headers = {"Authorization": f"Bearer {api_key}"}
    url = get_huggingface_url(endpoint)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response
                
            elif 'estimated_time' in response.text:
                # Handle model loading
                estimated_time = response.json().get('estimated_time', 30)
                logger.info(f"Model loading, waiting {estimated_time} seconds...")
                time.sleep(estimated_time)
                continue
                
            else:
                raise Exception(f"HuggingFace API error: {response.text}")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

def numpy_int64_presenter(dumper, data):
    return dumper.represent_int(int(data))

EnhancedYAMLDumper.add_representer(np.int64, numpy_int64_presenter)

def dump_yaml(data, file_path):
    """
    Safely dumps a dictionary to a YAML file with custom formatting.
    Converts any numpy.int64 values to int to avoid YAML serialization errors.
    Uses multi-line string representation for better readability.
    """
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Convert numpy types in the entire data structure
    data = yaml.safe_load(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    with open(file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, Dumper=EnhancedYAMLDumper, default_flow_style=False, 
                  sort_keys=False, allow_unicode=True, width=1000, indent=2)


def save_combo_settings(settings_dict, combo_presets_dir):
    """Save combo settings to the AutoCombo directory."""
    try:
        os.makedirs(combo_presets_dir, exist_ok=True)
        settings_path = os.path.join(combo_presets_dir, 'combo_settings.yaml')
        
        with open(settings_path, 'w') as f:
            yaml.safe_dump(settings_dict, f)
        logger.info(f"Saved combo settings to {settings_path}")
        return settings_dict
    except Exception as e:
        logger.error(f"Error saving combo settings: {str(e)}")
        return None

def load_combo_settings(combo_presets_dir):
    """Load combo settings from the AutoCombo directory."""
    try:
        settings_path = os.path.join(combo_presets_dir, 'combo_settings.yaml')
        
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
                logger.info(f"Loaded combo settings from {settings_path}")
                return settings
        else:
            logger.warning(f"Combo settings file not found at {settings_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading combo settings: {str(e)}")
        return {}

def create_settings_from_ui(ui_settings):
    """
    Create settings.yaml from UI settings with proper type conversion.
    Handles UI values that may be boolean or string.
    """
    import json

    def convert_to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        return bool(value)

    # Load profiles
    profiles_path = os.path.join(
        folder_paths.base_path,
        "custom_nodes",
        "ComfyUI-IF_LLM",
        "IF_AI",
        "presets",
        "profiles.json"
    )
    
    with open(profiles_path, 'r') as f:
        profiles = json.load(f)

    profile_name = ui_settings.get('profile', 'IF_PromptMKR')
    profile_content = profiles.get(profile_name, {}).get('instruction', '')

    # If 'prime_directives' is empty, use the profile content
    prime_directives = ui_settings.get('prime_directives')
    if not prime_directives or prime_directives in (None, '', 'None'):
        prime_directives = profile_content

    settings = {
        'base_ip': str(ui_settings.get('base_ip', 'localhost')),
        'port': str(ui_settings.get('port', '11434')),
        'user_prompt': str(ui_settings.get('user_prompt', 'Who helped Safiro infiltrate the Zaltar Organisation?')),
        'llm_provider': str(ui_settings.get('llm_provider', 'ollama')),
        'llm_model': str(ui_settings.get('llm_model', 'llama3.1:latest')),
        'prime_directives': prime_directives,
        'temperature': float(ui_settings.get('temperature', 0.7)),
        'max_tokens': int(ui_settings.get('max_tokens', 2048)),
        'stop_string': None if ui_settings.get('stop_string') in (None, 'None') else str(ui_settings.get('stop_string')),
        'keep_alive': convert_to_bool(ui_settings.get('keep_alive', False)),
        'clear_history': convert_to_bool(ui_settings.get('clear_history', False)),
        'history_steps': int(ui_settings.get('history_steps', 10)),
        'top_k': int(ui_settings.get('top_k', 40)),
        'top_p': float(ui_settings.get('top_p', 0.9)),
        'repeat_penalty': float(ui_settings.get('repeat_penalty', 1.2)),
        'seed': None if ui_settings.get('seed') in (None, 'None') else int(ui_settings.get('seed')),
        'external_api_key': str(ui_settings.get('external_api_key', '')),
        'random': convert_to_bool(ui_settings.get('random', False)),
        'aspect_ratio': str(ui_settings.get('aspect_ratio', '16:9')),
        'auto_combo': convert_to_bool(ui_settings.get('auto_combo', False)),
        'precision': str(ui_settings.get('precision', 'fp16')),
        'attention': str(ui_settings.get('attention', 'sdpa')),
        'batch_count': int(ui_settings.get('batch_count', 4)),
        'strategy': str(ui_settings.get('strategy', 'normal')),
        'profile': profile_name  # Include profile name
    }
    return settings

def format_response(self, response):
        """
        Format the response by adding appropriate line breaks and paragraph separations.
        """
        paragraphs = re.split(r"\n{2,}", response)

        formatted_paragraphs = []
        for para in paragraphs:
            if "```" in para:
                parts = para.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # This is a code block
                        parts[i] = f"\n```\n{part.strip()}\n```\n"
                para = "".join(parts)
            else:
                para = para.replace(". ", ".\n")

            formatted_paragraphs.append(para.strip())

        return "\n\n".join(formatted_paragraphs)

def print_available_models():
    """Print available models for each supported API engine"""
    
    # Test API key - using a dummy value since we'll mostly see fallback models
    test_api_key = "1234"
    
    # List of all supported engines
    engines = [
        "ollama",
        "huggingface",
        "deepseek", 
        "lmstudio",
        "textgen",
        "kobold",
        "llamacpp",
        "vllm",
        "openai",
        "xai",
        "mistral",
        "groq",
        "anthropic",
        "gemini",
        "sentence_transformers",
        "transformers"
    ]
    
    print("\n=== Available Models by Engine ===\n")
    
    for engine in engines:
        print(f"\n{engine.upper()} Models:")
        print("-" * (len(engine) + 8))
        
        try:
            # Get models for the current engine
            models = get_models(engine, "localhost", "11434", test_api_key)
            
            if models:
                # Print each model with an index
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
            else:
                print("No models available or engine requires valid API key/connection")
                
        except Exception as e:
            print(f"Error fetching models: {str(e)}")
            
        print() # Add blank line between engines

# Usage example:
if __name__ == "__main__":
    print_available_models()
