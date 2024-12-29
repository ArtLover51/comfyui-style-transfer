import os
import sys
import json
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
import numpy as np
import cv2

sys.path.append("ComfyUI")
import folder_paths
import execution
from nodes import init_custom_nodes

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        init_custom_nodes()
        
        # Set up model paths
        folder_paths.add_model_folder("checkpoints", "models/checkpoints")
        folder_paths.add_model_folder("controlnet", "models/controlnet")
        folder_paths.add_model_folder("ipadapter", "models/ipadapter")
        folder_paths.add_model_folder("upscale_models", "models/upscale")

    def predict(
        self,
        image: Path = Input(description="Input image to transform"),
        style_image: Path = Input(description="Style reference image"),
        prompt: str = Input(
            description="Positive prompt",
            default="Create an impressionistic painting"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="blurry, noisy, messy, glitch, distorted"
        ),
        structure_depth: float = Input(
            description="Structure preservation strength (0-1)",
            default=0.75,
            ge=0,
            le=1
        ),
        denoise_strength: float = Input(
            description="Denoising strength (0-1)",
            default=0.65,
            ge=0,
            le=1
        )
    ) -> Path:
        """Run style transfer on an input image"""
        
        # Load workflow template
        with open('workflow.json', 'r') as f:
            workflow = json.load(f)

        # Update workflow nodes with input parameters
        for node in workflow['nodes']:
            if node['type'] == 'LoadImage':
                if node['title'] == 'Style Image':
                    node['inputs'][0]['image'] = str(style_image)
                elif node['title'] == 'Input Image':
                    node['inputs'][0]['image'] = str(image)
            
            elif node['type'] == 'CLIPTextEncode':
                if 'positive' in node['title'].lower():
                    node['inputs'][0]['text'] = prompt
                elif 'negative' in node['title'].lower():
                    node['inputs'][0]['text'] = negative_prompt
            
            elif node['type'] == 'ControlNetApplyAdvanced':
                if 'depth' in node['title'].lower():
                    node['inputs'][0]['strength'] = structure_depth
                elif 'canny' in node['title'].lower():
                    node['inputs'][0]['strength'] = 0.75
            
            elif node['type'] == 'KSampler':
                node['inputs'][6]['denoise'] = denoise_strength

        # Execute workflow
        graph = execution.PromptExecutor(workflow)
        output_images = graph.execute()
        
        # Save and return the final image
        final_image = output_images[-1][0]
        output_path = Path(os.path.join(os.getcwd(), "output.png"))
        final_image.save(str(output_path))
        
        return output_path
