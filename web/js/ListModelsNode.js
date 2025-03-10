import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "ImpactFrames.ListModelsNode",
    
    async setup() {
        // Register for model list updates
        api.addEventListener("impact_frames_model_list_update", this.handleModelListUpdate);
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only customize our specific node
        if (nodeType.comfyClass === "ListModelsNode") {
            // Store the original onExecuted method
            const origExecuted = nodeType.prototype.onExecuted;
            
            // Override the onExecuted method
            nodeType.prototype.onExecuted = function(message) {
                // Call the original method first
                const result = origExecuted?.apply(this, arguments);
                
                // Handle the model list display in a better format
                if (this.widgets && message?.output?.model_list) {
                    const modelList = message.output.model_list;
                    
                    // Find or create an output widget to display the model list
                    let outputWidget = this.widgets.find(w => w.name === "models_output");
                    if (!outputWidget) {
                        outputWidget = this.addWidget("customtext", "models_output", "", (v) => {});
                        outputWidget.inputEl.style.height = "200px";
                        outputWidget.inputEl.readOnly = true;
                        outputWidget.inputEl.style.fontFamily = "monospace";
                        outputWidget.inputEl.style.fontSize = "12px";
                        outputWidget.inputEl.style.overflow = "auto";
                    }
                    
                    // Set the model list content
                    outputWidget.value = modelList;
                    outputWidget.inputEl.value = modelList;
                    
                    // Add a copy button if it doesn't exist
                    if (!this.copyButton) {
                        this.copyButton = document.createElement("button");
                        this.copyButton.innerText = "Copy Models List";
                        this.copyButton.style.marginLeft = "5px";
                        this.copyButton.addEventListener("click", () => {
                            navigator.clipboard.writeText(modelList)
                                .then(() => {
                                    // Show a temporary success message
                                    const originalText = this.copyButton.innerText;
                                    this.copyButton.innerText = "Copied!";
                                    setTimeout(() => {
                                        this.copyButton.innerText = originalText;
                                    }, 1500);
                                })
                                .catch(err => console.error("Copy failed:", err));
                        });
                        
                        // Add the button next to the last widget
                        if (this.widgets.length > 0) {
                            const lastWidget = this.widgets[this.widgets.length - 1];
                            if (lastWidget.inputEl && lastWidget.inputEl.parentElement) {
                                lastWidget.inputEl.parentElement.appendChild(this.copyButton);
                            }
                        }
                    }
                    
                    // Refresh the UI
                    this.setDirtyCanvas(true, true);
                }
                
                return result;
            };
            
            // Add custom menu options to the node
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                getExtraMenuOptions?.apply(this, arguments);
                
                options.push(
                    null, // Add a divider
                    {
                        content: "Save Model List to File",
                        callback: () => {
                            // Find the output widget
                            const outputWidget = this.widgets.find(w => w.name === "models_output");
                            if (outputWidget && outputWidget.value) {
                                // Create a temporary download link
                                const providerWidget = this.widgets.find(w => w.name === "llm_provider");
                                const provider = providerWidget ? providerWidget.value : "models";
                                
                                const blob = new Blob([outputWidget.value], { type: 'text/plain' });
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `${provider}_models.txt`;
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                URL.revokeObjectURL(url);
                            }
                        }
                    }
                );
            };
        }
    },
    
    // Handle model list updates from the server
    handleModelListUpdate(event) {
        const { nodeId, modelList } = event.detail;
        
        // Find the node
        const node = app.graph.getNodeById(nodeId);
        if (node) {
            // Update the node's output widget
            let outputWidget = node.widgets.find(w => w.name === "models_output");
            if (outputWidget) {
                outputWidget.value = modelList;
                outputWidget.inputEl.value = modelList;
                node.setDirtyCanvas(true, true);
            }
        }
    }
});
