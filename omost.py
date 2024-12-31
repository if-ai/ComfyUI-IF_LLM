import sys
import os
from typing import Dict, Any, List
import json
import asyncio

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the 'lib_omost' directory to sys.path
lib_omost_dir = os.path.join(current_dir, 'lib_omost')
if lib_omost_dir not in sys.path:
    sys.path.append(lib_omost_dir)

from lib_omost.canvas import Canvas as OmostCanvas, OmostCanvasCondition, system_prompt

class OmostTool:
    def __init__(self, name, description, system_prompt):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.output_type = "canvas_conditioning"
        #print(f"OmostTool initialized with system prompt: {self.system_prompt}")

    async def execute(self, args) -> Dict[str, Any]:
        #print("DEBUG(omost.py): Entered OmostTool.execute with args:", args)
        #print(f"OmostTool execute method called with args: {args}")
        prompt = args.get('input', '')
        llm_response = args.get('llm_response', '')
        
        #print(f"Prompt: {prompt}")
        #print(f"LLM Response: {llm_response}")
        
        try:
            canvas = OmostCanvas.from_bot_response(llm_response)
            canvas_conditioning = canvas.process()
            # Ensure canvas_conditioning is a flat list of dicts
            if (
                isinstance(canvas_conditioning, list)
                and len(canvas_conditioning) == 1
                and isinstance(canvas_conditioning[0], list)
            ):
                # Flatten once
                canvas_conditioning = canvas_conditioning[0]

            print("Canvas processed successfully")
            
            result = {
                self.output_type: canvas_conditioning,
                "prompt": prompt,
                "llm_response": llm_response
            }
        except Exception as e:
            print(f"Error processing canvas: {str(e)}")
            result = {
                "error": str(e),
                "prompt": prompt,
                "llm_response": llm_response
            }
        
        #print(f"OmostTool execute method returning: {result}")
        return result

async def omost_function(args: Dict[str, Any]) -> Dict[str, Any]:
    tool = OmostTool(args['name'], args['description'], args['system_prompt'])
    result = await tool.execute(args)
    return result

