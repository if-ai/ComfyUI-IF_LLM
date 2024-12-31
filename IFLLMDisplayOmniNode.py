# IFDisplayOmniNode.py
import json

class IFDisplayOmni:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {"omni_input": ("OMNI", {})},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("OMOST_CANVAS_CONDITIONING", "STRING")
    RETURN_NAMES = ("canvas_conditioning", "text_output")
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "display_omni"
    CATEGORY = "ImpactFrames💥🎞️/IF_LLM"

    def flatten_conditioning(self, conditioning):
        """Ensure conditioning is a flat list of dictionaries"""
        if not conditioning:
            return [{
                "color": [1.0, 1.0, 1.0],
                "prefixes": [""],
                "suffixes": [""],
                "rect": [0, 1, 0, 1]
            }]
        
        # Handle nested lists
        if isinstance(conditioning, list):
            if len(conditioning) == 1 and isinstance(conditioning[0], list):
                return self.flatten_conditioning(conditioning[0])
            
            # Ensure all items are dictionaries with required keys
            flattened = []
            for item in conditioning:
                if isinstance(item, list):
                    flattened.extend(self.flatten_conditioning(item))
                elif isinstance(item, dict):
                    # Ensure dictionary has required keys
                    item.setdefault("color", [1.0, 1.0, 1.0])
                    item.setdefault("prefixes", [""])
                    item.setdefault("suffixes", [""])
                    item.setdefault("rect", [0, 1, 0, 1])
                    flattened.append(item)
            return flattened
            
        return [{
            "color": [1.0, 1.0, 1.0],
            "prefixes": [""],
            "suffixes": [""],
            "rect": [0, 1, 0, 1]
        }]

    def extract_text_content(self, val):
        """Extract textual content from various input types"""
        if isinstance(val, dict):
            # Try to get text content from different possible keys
            return (val.get("llm_response") or 
                   val.get("error") or 
                   val.get("text") or 
                   val.get("content") or 
                   str(val))
        elif isinstance(val, list):
            # For lists, try to extract text from each item
            texts = []
            for item in val:
                if isinstance(item, dict):
                    text = self.extract_text_content(item)
                    if text:
                        texts.append(text)
            return "\n".join(texts) if texts else str(val)
        elif isinstance(val, str):
            return val
        return str(val)

    def display_omni(self, unique_id=None, extra_pnginfo=None, **kwargs):
        values = []
        canvas_conditioning = None
        text_output = ""
        all_text = []

        if "omni_input" in kwargs:
            for val in kwargs['omni_input']:
                try:
                    if isinstance(val, dict):
                        if "conditionings" in val:
                            canvas_conditioning = val["conditionings"]
                            # Extract text content from the dict
                            extracted_text = self.extract_text_content(val)
                            all_text.append(extracted_text)
                            values.append(extracted_text)
                        elif "canvas_conditioning" in val:
                            canvas_conditioning = val["canvas_conditioning"]
                            extracted_text = self.extract_text_content(val)
                            all_text.append(extracted_text)
                            values.append(extracted_text)
                        else:
                            # Handle other dictionary types
                            extracted_text = self.extract_text_content(val)
                            all_text.append(extracted_text)
                            values.append(extracted_text)

                    elif isinstance(val, list):
                        canvas_conditioning = val
                        extracted_text = self.extract_text_content(val)
                        all_text.append(extracted_text)
                        values.append(extracted_text)

                    elif isinstance(val, str):
                        all_text.append(val)
                        values.append(val)
                        
                    else:
                        text = str(val)
                        all_text.append(text)
                        values.append(text)
                        
                except Exception as e:
                    error_text = f"Error processing omni input: {str(e)}"
                    print(error_text)
                    all_text.append(error_text)
                    values.append(str(val))

        # Update workflow info if available
        if unique_id is not None and extra_pnginfo is not None:
            if isinstance(extra_pnginfo, list) and len(extra_pnginfo) > 0:
                extra_pnginfo = extra_pnginfo[0]
            
            if isinstance(extra_pnginfo, dict) and "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
                if node:
                    node["widgets_values"] = [values]

        # Ensure canvas_conditioning is a flattened list of dictionaries
        canvas_conditioning = self.flatten_conditioning(canvas_conditioning)
        
        # Combine all collected text into final text output
        text_output = "\n".join(all_text) if all_text else ""

        return {
            "ui": {"text": values},
            "result": (canvas_conditioning, text_output)
        }