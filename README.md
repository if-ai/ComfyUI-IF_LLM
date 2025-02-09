
# ComfyUI-IF_AI_LLM

################# ATENTION ####################

   *It Might comflict with IF_AI_tools so if you have 
   it installed please remove it before installing IF_LLM 
   I am working on adding this tools to IF_AI_tools 
   so you only need one or the other*
   
###############################################


# Video

[![Video Thumbnail](https://github.com/user-attachments/assets/7430c137-9193-48dd-be34-fddbb2cd0387)](https://youtu.be/0sR4hu98pDo?si=EhF24ugy7RpLvUjV)


Lighter version of ComfyUI-IF_AI_tools is a set of custom nodes to Run Local and API LLMs and LMMs, supports Ollama, LlamaCPP LMstudio, Koboldcpp, TextGen, Transformers or via APIs Anthropic, Groq, OpenAI, Google Gemini, Mistral, xAI and create your own profiles (SystemPrompts) with custom presets and muchmore

![thorium_HQhmKkuczP](https://github.com/user-attachments/assets/547f1096-fb5e-4249-95bd-1f6920788aa2)


### Install Ollama

You can technically use any LLM API that you want, but for the best expirience install Ollama and set it up.
- Visit [ollama.com](https://ollama.com) for more information.

To install Ollama models just open CMD or any terminal and type the run command follow by the model name such as
```powershell
ollama run llama3.2-vision
```
If you want to use omost 
```bash
ollama run impactframes/dolphin_llama3_omost
```
if you need a good smol model
```bash
ollama run ollama run llama3.2
```

Optionally Set enviromnet variables for any of your favourite LLM API keys "XAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "OPENAI_API_KEY" or "GROQ_API_KEY" with those names or otherwise
it won't pick it up you can also use .env file to store your keys

## Features
_[NEW]_ xAI Grok Vision, Mistral, Google Gemini exp 114, Anthropic 3.5 Haiku, OpenAI 01 preview
_[NEW]_ Wildcard System
_[NEW]_ Local Models Koboldcpp, TextGen, LlamaCPP, LMstudio, Ollama
_[NEW]_ Auto prompts auto generation for Image Prompt Maker runs jobs on batches automatically
_[NEW]_ Image generation with IF_PROMPTImaGEN via Dalle3 
_[NEW]_ Endpoints xAI, Transformers,
_[NEW]_ IF_profiles System Prompts with Reasoning/Reflection/Reward Templates and custom presets
_[NEW]_ WF such as GGUF and FluxRedux

- Gemini, Groq, Mistral, OpenAI, Anthropic, Google, xAI, Transformers, Koboldcpp, TextGen, LlamaCPP, LMstudio, Ollama 
- Omost_tool the first tool 
- Vision Models Haiku/GPT4oMini?Geminiflash/Qwen2-VL 
- [Ollama-Omost]https://ollama.com/impactframes/dolphin_llama3_omost can be 2x to 3x faster than other Omost Models
LLama3 and Phi3 IF_AI Prompt mkr models released
![thorium_XXW2qsjUp0](https://github.com/user-attachments/assets/89bb5e3f-f103-4c64-b086-ed6194747f9b)


`ollama run impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest`

`ollama run impactframes/ifai_promptmkr_dolphin_phi3:latest`

https://huggingface.co/impactframes/llama3_if_ai_sdpromptmkr_q4km

https://huggingface.co/impactframes/ifai_promptmkr_dolphin_phi3_gguf


## Installation
1. Open the manager search for IF_LLM and install

### Install ComfyUI-IF_AI_ImaGenPromptMaker -hardest way
   
1. Navigate to your ComfyUI `custom_nodes` folder, type `CMD` on the address bar to open a command prompt,
   and run the following command to clone the repository:
   ```bash
      git clone https://github.com/if-ai/ComfyUI-IF_LLM.git
      ```
OR
1. In ComfyUI protable version just dounle click `embedded_install.bat` or  type `CMD` on the address bar on the newly created `custom_nodes\ComfyUI-IF_LLM` folder type 
   ```bash
      H:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt
      ```
   replace `C:\` for your Drive letter where you have the ComfyUI_windows_portable directory

2. On custom environment activate the environment and move to the newly created ComfyUI-IF_LLM
   ```bash
      cd ComfyUI-IF_LLM
      python -m pip install -r requirements.txt
      ```
![thorium_59oWhA71y7](https://github.com/user-attachments/assets/e9641052-4838-4ee3-91c4-7e02190e9064)

## Related Tools
- [IF_prompt_MKR](https://github.com/if-ai/IF_PROMPTImaGEN) 
-  A similar tool available for Stable Diffusion WebUI

## Videos

None yet

## Example using normal Model
ancient Megastructure, small lone figure 


## TODO
- [ ] IMPROVED PROFILES
- [ ] OMNIGEN
- [ ] QWENFLUX
- [ ] VIDEOGEN
- [ ] AUDIOGEN

## Support
If you find this tool useful, please consider supporting my work by:
- Starring the repository on GitHub: [ComfyUI-IF_AI_tools](https://github.com/if-ai/ComfyUI-IF_AI_tools)
- Subscribing to my YouTube channel: [Impact Frames](https://youtube.com/@impactframes?si=DrBu3tOAC2-YbEvc)
- Follow me on X: [Impact Frames X](https://x.com/impactframesX)
Thank You!

<img src="https://count.getloli.com/get/@IFAIPROMPTImaGEN_comfy?theme=moebooru" alt=":IFAIPROMPTImaGEN_comfy" />
