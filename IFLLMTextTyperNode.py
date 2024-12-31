class IFTextTyper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "output_text"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️/IF_LLM"

    def output_text(self, text):
        return (text,)

NODE_CLASS_MAPPINGS = {
    "IF_LLM_TextTyper": IFTextTyper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_LLM_TextTyper": "IF Text Typer✍️"
}