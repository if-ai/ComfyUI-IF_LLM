
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


## Workflow Examples
You can try out these workflow examples directly in ComfyDeploy!

| Workflow | Try It |
|--------------|---------|
|[CD_HYVid_I2V_&_T2V_Native_IFLLM](workflows/CD_HYVid_I2V_%26_T2V_Native_IFLLM.json)|[![Try CD_HYVid_I2V_&_T2V_Native_IFLLM](https://beta.app.comfydeploy.com/button)](https://beta.app.comfydeploy.com/home?gpu=L40S&comfyui_version=a7fe0a94dee08754f97b0171e15c1f2271aa37be&timeout=15&nodes=if-ai%2FComfyUI-IF_LLM%40c80e379%2Crgthree%2Frgthree-comfy%405d771b8%2CJonseed%2FComfyUI-Detail-Daemon%4090e703d%2Ckijai%2FComfyUI-KJNodes%40a22b269%2Ccubiq%2FComfyUI_essentials%4033ff89f%2CTinyTerra%2FComfyUI_tinyterraNodes%40b292f8e%2Cchengzeyi%2FComfy-WaveSpeed%403db162b%2CTTPlanetPig%2FComfyui_TTP_Toolset%406dd3f35%2Ckijai%2FComfyUI-HunyuanVideoWrapper%409f50ed1%2CKosinkadink%2FComfyUI-VideoHelperSuite%403bfbd99%2CFannovel16%2FComfyUI-Frame-Interpolation%40c336f71%2Cfacok%2FComfyUI-HunyuanVideoMultiLora%407e3e344%2Ccity96%2FComfyUI_ExtraModels%4092f556e%2Cblepping%2FComfyUI-bleh%40850f840%2CjamesWalker55%2Fcomfyui-various%4036454f9&workflowLink=https%3A%2F%2Fraw.githubusercontent.com%2Fif-ai%2FComfyUI-IF_LLM%2Fmain%2Fworkflows%2FCD_HYVid_I2V_%26_T2V_Native_IFLLM.json)|
|[CD_HYVid_I2V_&_T2V_i2VLora_Native](workflows/CD_HYVid_I2V_%26_T2V_i2VLora_Native.json)|[![Try CD_HYVid_I2V_&_T2V_i2VLora_Native](https://beta.app.comfydeploy.com/button)](https://beta.app.comfydeploy.com/home?gpu=l40s&comfyui_version=a7fe0a94dee08754f97b0171e15c1f2271aa37be&timeout=15&nodes=if-ai/ComfyUI-IF_LLM%40c80e379%2Crgthree/rgthree-comfy%405d771b8%2CJonseed/ComfyUI-Detail-Daemon%4090e703d%2Ckijai/ComfyUI-KJNodes%40a22b269%2Ccubiq/ComfyUI_essentials%4033ff89f%2CTinyTerra/ComfyUI_tinyterraNodes%40b292f8e%2Cchengzeyi/Comfy-WaveSpeed%403db162b%2CTTPlanetPig/Comfyui_TTP_Toolset%406dd3f35%2Ckijai/ComfyUI-HunyuanVideoWrapper%409f50ed1%2CKosinkadink/ComfyUI-VideoHelperSuite%403bfbd99%2CFannovel16/ComfyUI-Frame-Interpolation%40c336f71%2Cfacok/ComfyUI-HunyuanVideoMultiLora%407e3e344&workflowLink=https%3A//raw.githubusercontent.com/if-ai/ComfyUI-IF_LLM/main/workflows/CD_HYVid_I2V_%26_T2V_i2VLora_Native.json)|
|[CD_HYVid_I2V_Lora_KjWrapper](workflows/CD_HYVid_I2V_Lora_KjWrapper.json)|[![Try CD_HYVid_I2V_Lora_KjWrapper](https://beta.app.comfydeploy.com/button)](https://beta.app.comfydeploy.com/home?gpu=l40s&comfyui_version=a7fe0a94dee08754f97b0171e15c1f2271aa37be&timeout=15&nodes=if-ai/ComfyUI-IF_LLM%40c80e379%2Crgthree/rgthree-comfy%405d771b8%2CJonseed/ComfyUI-Detail-Daemon%4090e703d%2Ckijai/ComfyUI-KJNodes%40a22b269%2Ccubiq/ComfyUI_essentials%4033ff89f%2CTinyTerra/ComfyUI_tinyterraNodes%40b292f8e%2Cchengzeyi/Comfy-WaveSpeed%403db162b%2CTTPlanetPig/Comfyui_TTP_Toolset%406dd3f35%2Ckijai/ComfyUI-HunyuanVideoWrapper%409f50ed1%2CKosinkadink/ComfyUI-VideoHelperSuite%403bfbd99%2CFannovel16/ComfyUI-Frame-Interpolation%40c336f71%2Cfacok/ComfyUI-HunyuanVideoMultiLora%407e3e344&workflowLink=https%3A//raw.githubusercontent.com/if-ai/ComfyUI-IF_LLM/main/workflows/CD_HYVid_I2V_Lora_KjWrapper.json)|

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
