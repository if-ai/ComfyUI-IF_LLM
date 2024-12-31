import sys
import logging
from typing import Optional, Union, List

# Initialize logger
logger = logging.getLogger(__name__)

class IFDisplayText:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {        
                "text": ("STRING", {"forceInput": True}),
                "select": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": sys.maxsize,  # No practical upper limit
                    "step": 1,
                    "tooltip": "Select which line to output (cycles through available lines)"
                }),     
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "text_list", "count", "selected")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "display_text"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️/IF_LLM"
    
    def display_text(self, text: Optional[Union[str, List[str]]], select):
        if text is None:
            logger.error("Received None for text input in display_text.")
            return "", [], 0, ""
    
        print("==================")
        print("IF_AI_tool_output:")
        print("==================")
        print(text)
        
        # Initialize variables
        text_list = []
    
        if isinstance(text, list):
            # Handle list of strings
            for idx, item in enumerate(text):
                if isinstance(item, str):
                    lines = [line.strip() for line in item.split('\n') if line.strip()]
                    text_list.extend(lines)
                else:
                    logger.warning(f"Expected string in text list at index {idx}, but got {type(item)}")
        elif isinstance(text, str):
            # Handle single string
            text_list = [line.strip() for line in text.split('\n') if line.strip()]
        else:
            logger.error(f"Unexpected type for text: {type(text)}")
            return "", [], 0, ""
    
        count = len(text_list)
        
        # Select line using modulo to handle cycling
        if count == 0:
            selected = text if isinstance(text, str) else ""
        else:
            selected = text_list[select % count]
        
        # Prepare UI update
        if isinstance(text, list):
            ui_text = text  # Pass the list directly
        else:
            ui_text = [text]  # Wrap single string in a list
        
        # Return both UI update and the multiple outputs
        return {
            "ui": {"string": ui_text}, 
            "result": (
                text,        # complete text (string or list)
                text_list,   # list of individual lines as separate string outputs
                count,       # number of lines
                selected     # selected line based on select input
            )
        }

NODE_CLASS_MAPPINGS = {"IF_LLM_DisplayText": IFDisplayText}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_LLM_DisplayText": "IF Display Text📟"}


