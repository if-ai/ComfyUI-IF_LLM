import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import folder_paths
import folder_paths

# Then import your other modules
from .IFLLMNode import IFLLM
from .IFDisplayTextWildcardNode import IFDisplayTextWildcard
from .IFLLMSaveTextNode import IFSaveText
from .IFLLMDisplayTextNode import IFDisplayText
from .IFLLMDisplayOmniNode import IFDisplayOmni
from .IFLLMTextTyperNode import IFTextTyper
from .IFLLMJoinTextNode import IFJoinText
from .IFLLMLoadImagesNodeS import IFLoadImagess
from .send_request import *


'''# Unified omost import handling
try:
    if "omost" not in sys.modules:  # Check if already imported
        try:
            from .omost import omost_function  # Try relative import first
            print("Successfully imported omost_function from current directory")
        except ImportError:
            # If relative import fails, try from parent directory
            parent_dir = os.path.dirname(current_dir)
            parent_dir_name = os.path.basename(parent_dir)
            if parent_dir_name == 'ComfyUI-IF_LLM':
                sys.path.insert(0, parent_dir)
                from omost import omost_function
                print(f"Successfully imported omost_function from {parent_dir}/omost.py")
    else:
        omost_function = sys.modules["omost"].omost_function
        print("Using already imported omost_function")
except ImportError as e:
    print(f"Error importing omost_function: {e}")
    print(f"Current sys.path: {sys.path}")
    raise'''

class OmniType(str):
    """A special string type that acts as a wildcard for universal input/output. 
       It always evaluates as equal in comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False
    
OMNI = OmniType("*")
                       
NODE_CLASS_MAPPINGS = {
    "IF_LLM": IFLLM,
    "IF_LLM_SaveText": IFSaveText,
    "IF_LLM_DisplayText": IFDisplayText,
    "IF_LLM_DisplayTextWildcard": IFDisplayTextWildcard,
    "IF_LLM_DisplayOmni": IFDisplayOmni,
    "IF_LLM_TextTyper": IFTextTyper,
    "IF_LLM_JoinText": IFJoinText,
    "IF_LLM_LoadImagesS": IFLoadImagess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_LLM": "IF LLM🎨",
    "IF_LLM_SaveText": "IF Save Text📝",
    "IF_LLM_DisplayText": "IF Display Text📟",
    "IF_LLM_DisplayTextWildcard": "IF Display Text Wildcard📟",
    "IF_LLM_DisplayOmni": "IF Display Omni🔍",
    "IF_LLM_TextTyper": "IF Text Typer✍️",
    "IF_LLM_JoinText": "IF Join Text 📝",
    "IF_LLM_LoadImagesS": "IF LLM Load Images S 🖼️"
}

WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY", 
    "omost_function"
    ]
