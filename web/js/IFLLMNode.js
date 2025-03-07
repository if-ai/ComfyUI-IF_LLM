//IFLLMNode.js
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.IFLLMNode",
    
    async setup() {
        let attempts = 0;
        const maxAttempts = 10;
        const waitTime = 1000;

        while ((!app.ui?.settings?.store || !app.api) && attempts < maxAttempts) {
            console.log(`Attempt ${attempts + 1}/${maxAttempts}: Waiting for UI and API to initialize...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            attempts++;
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_LLM") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Single onNodeCreated implementation that combines all functionality
            nodeType.prototype.onNodeCreated = function() {
                // Call original if it exists
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }

                const self = this;

                // Add settings button
                const saveComboSettings = this.addWidget("button", "Store Auto Prompt", null, () => {
                    const settings = this.getNodeComboSettings();
                    
                    fetch("/IF_LLM/save_combo_settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(settings)
                    })
                    .then(response => response.json())
                    .then(result => {
                        if (result.status === "success") {
                            alert("Combo settings saved successfully!");
                        } else {
                            alert("Error saving settings: " + result.message);
                        }
                    })
                    .catch(error => {
                        console.error("Error saving combo settings:", error);
                        alert("Error saving settings: " + error.message);
                    });
                });
                
                // Configure button styling
                saveComboSettings.serialize = false;
                
                // Add LLM model update functionality
                const updateLLMModels = async () => {
                    const llmProviderWidget = this.widgets.find((w) => w.name === "llm_provider");
                    const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
                    const portWidget = this.widgets.find((w) => w.name === "port");
                    const llmModelWidget = this.widgets.find((w) => w.name === "llm_model");
                    const externalApiKeyWidget = this.widgets.find((w) => w.name === "external_api_key");

                    if (llmProviderWidget && baseIpWidget && portWidget && llmModelWidget) {
                        try {
                            const response = await fetch("/IF_LLM/get_llm_models", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    llm_provider: llmProviderWidget.value,
                                    base_ip: baseIpWidget.value,
                                    port: portWidget.value,
                                    external_api_key: externalApiKeyWidget?.value || ""
                                })
                            });

                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }

                            const models = await response.json();
                            console.log("Fetched models:", models);

                            if (Array.isArray(models) && models.length > 0) {
                                llmModelWidget.options.values = models;
                                llmModelWidget.value = models[0];
                                this.setDirtyCanvas(true, true);
                            } else {
                                throw new Error("No models available");
                            }
                        } catch (error) {
                            console.error("Error updating models:", error);
                            
                            // Fallback models
                            const fallbackModels = {
                                openai: [
                                    // Vision Language Models (VLM)
                                    "meta-llama/Llama-3.2-11B-Vision-Instruct",
                                    "Qwen/Qwen2-VL-7B-Chat",
                                    "Qwen/Qwen2-VL-7B",
                                    "Qwen/Qwen2-VL-2B-Chat",
                                    "Qwen/Qwen2-VL-2B",
                                    "Qwen/Qwen2-VL-7B-Instruct",
                                    "Qwen/Qwen2-VL-2B-Instruct",
                                    "microsoft/phi-2",
                                    "HuggingFaceH4/zephyr-7b-beta",
                                    
                                    // Text to Image Models
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
                                    
                                    // Image to Image Models
                                    "timbrooks/instruct-pix2pix",
                                    "lambdalabs/sd-image-variations-diffusers",
                                    "diffusers/controlnet-canny-sdxl-1.0",
                                    
                                    // Specialized Models
                                    "kandinsky-community/kandinsky-3",
                                    "stabilityai/stable-cascade",
                                    "dataautogpt3/OpenDalle3",
                                    "ByteDance/SDXL-Lightning",
                                    
                                    // ControlNet Models
                                    "lllyasviel/control_v11p_sd15_canny",
                                    "lllyasviel/control_v11p_sd15_openpose",
                                    "lllyasviel/control_v11p_sd15_depth",
                                    
                                    // Text Feature Extraction
                                    "sentence-transformers/all-MiniLM-L6-v2",
                                    "sentence-transformers/all-mpnet-base-v2",
                                    
                                    // Image Feature Extraction  
                                    "openai/clip-vit-base-patch32",
                                    "openai/clip-vit-large-patch14",
                                    
                                    // Text Classification
                                    "distilbert-base-uncased-finetuned-sst-2-english",
                                    "roberta-base-openai-detector",
                                    
                                    // Text Generation
                                    "gpt2",
                                    "facebook/opt-350m",
                                    
                                    // Translation 
                                    "Helsinki-NLP/opus-mt-en-fr",
                                    "Helsinki-NLP/opus-mt-fr-en",
                                    
                                    // Question Answering
                                    "deepset/roberta-base-squad2",
                                    "distilbert-base-cased-distilled-squad"
                                ],
                                anthropic: [
                                    "claude-3-5-opus-latest",
                                    "claude-3-opus-20240229",
                                    "claude-3-5-sonnet-latest",
                                    "claude-3-5-sonnet-20240620",
                                    "claude-3-sonnet-20240229",
                                    "claude-3-haiku-20240307",
                                    "claude-3-5-haiku-latest",
                                    "claude-3-5-haiku-20241022"
                                ],
                                ollama: ["llava", "llava-v1.5-7b", "bakllava"],
                                groq: [
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
                                ],
                                gemini: [
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
                                ],
                                mistral: [
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
                                ],
                                deepseek: [
                                    "deepseek-reasoner",
                                    "deepseek-chat",
                                    "deepseek-coder"
                                ],
                                xai: [
                                    "grok-2",
                                    "grok-2-1212",
                                    "grok-2-latest",
                                    "grok-2-vision",
                                    "grok-2-vision-1212",
                                    "grok-2-vision-latest",
                                    "grok-beta",
                                    "grok-vision-beta"
                                ],
                                transformers: [
                                    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                                    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                                    "Qwen/QwQ-32B-AWQ",
                                    "Qwen/Qwen2.5-VL-3B-Instruct", 
                                    "Qwen/Qwen2.5-VL-7B-Instruct",
                                    "Qwen/Qwen2-72B-Instruct",
                                    "Qwen/Qwen2-7B-Instruct",
                                    "Qwen/Qwen2-VL-2B-Instruct",
                                    "Qwen/Qwen2-VL-7B-Instruct"
                                ]
                            };

                            const models = fallbackModels[llmProviderWidget.value] || ["No models available"];
                            llmModelWidget.options.values = models;
                            llmModelWidget.value = models[0];
                        }
                    }
                };

                // Node settings collection
                this.getNodeComboSettings = function() {
                    const getWidgetValue = (name) => {
                        const widget = this.widgets.find(w => w.name === name);
                        return widget ? widget.value : undefined;
                    };

                    return {
                        llm_provider: getWidgetValue('llm_provider'),
                        llm_model: getWidgetValue('llm_model'),
                        base_ip: getWidgetValue('base_ip'),
                        port: getWidgetValue('port'),
                        user_prompt: getWidgetValue('user_prompt'),
                        profile: getWidgetValue('profiles'),
                        prime_directives: getWidgetValue('prime_directives'),
                        temperature: getWidgetValue('temperature'),
                        max_tokens: getWidgetValue('max_tokens'),
                        stop_string: getWidgetValue('stop_string'),
                        keep_alive: getWidgetValue('keep_alive'),
                        top_k: getWidgetValue('top_k'),
                        top_p: getWidgetValue('top_p'),
                        repeat_penalty: getWidgetValue('repeat_penalty'),
                        seed: getWidgetValue('seed'),
                        external_api_key: getWidgetValue('external_api_key'),
                        random: getWidgetValue('random'),
                        precision: getWidgetValue('precision'),
                        attention: getWidgetValue('attention'),
                        aspect_ratio: getWidgetValue('aspect_ratio'),
                        batch_count: getWidgetValue('batch_count'),
                        strategy: getWidgetValue('strategy')
                    };
                };

                // Set up widget callbacks
                this.widgets.forEach(w => {
                    if (["llm_provider", "base_ip", "port", "external_api_key"].includes(w.name)) {
                        w.callback = updateLLMModels;
                    }
                });

                // Initial model update
                updateLLMModels();
            };

            // Add node preview handling
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                if (this.generated_prompt) {
                    const margin = 10;
                    const textX = this.pos[0] + margin;
                    const textY = this.pos[1] + this.size[1] + 20;
                    const maxWidth = this.size[0] - margin * 2;
                    
                    ctx.save();
                    ctx.font = "12px Arial";
                    ctx.fillStyle = "#CCC";
                    this.wrapText(ctx, this.generated_prompt, textX, textY, maxWidth, 16);
                    ctx.restore();
                }
            };

            // Add helper methods
            nodeType.prototype.wrapText = function(ctx, text, x, y, maxWidth, lineHeight) {
                const words = text.split(' ');
                let line = '';
                let posY = y;

                for (const word of words) {
                    const testLine = line + word + ' ';
                    const metrics = ctx.measureText(testLine);
                    const testWidth = metrics.width;

                    if (testWidth > maxWidth && line !== '') {
                        ctx.fillText(line, x, posY);
                        line = word + ' ';
                        posY += lineHeight;
                    } else {
                        line = testLine;
                    }
                }
                ctx.fillText(line, x, posY);
            };

            // Handle execution results
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                if (message?.generated_prompt) {
                    this.generated_prompt = message.generated_prompt;
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    }
});