from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import matplotlib.pyplot as plt
import base64
from pydantic import BaseModel

class SDBody(BaseModel):
    prompt: str
    depth: str
    image: str

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost:8080",  # Adjust the origin according to your Vue.js app's URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

def get_canny_image(control_image):
    low_threshold = 50
    high_threshold = 200

    control_image = cv2.Canny(control_image, low_threshold, high_threshold)
    control_image = control_image[:, :, None]
    control_image = np.concatenate([control_image, control_image, control_image], axis=2)
    control_image = Image.fromarray(control_image)
    return control_image

model_path = '/home/alex/stable-diffusion-webui/models/Stable-diffusion/photon_v1.safetensors'

control_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16).to('cuda')
control_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to('cuda')
generator = torch.Generator(device="cuda").manual_seed(0)

text2img_pipeline = StableDiffusionControlNetPipeline.from_single_file(
    model_path, controlnet=control_depth , safety_checker=None, torch_dtype=torch.float16
).to('cuda')
text2img_pipeline.scheduler = UniPCMultistepScheduler.from_config(text2img_pipeline.scheduler.config)
text2img_pipeline.enable_model_cpu_offload()

img2img_pipeline = StableDiffusionControlNetImg2ImgPipeline(
    vae=text2img_pipeline.vae,
    text_encoder=text2img_pipeline.text_encoder,
    tokenizer=text2img_pipeline.tokenizer,
    unet=text2img_pipeline.unet,
    scheduler=text2img_pipeline.scheduler,
    safety_checker=text2img_pipeline.safety_checker,
    feature_extractor=text2img_pipeline.feature_extractor,
    controlnet = control_canny
)
img2img_pipeline.enable_model_cpu_offload()

def base64_to_pil(base64_string):
    # Decode the Base64 string to binary data
    image_data = base64.b64decode(base64_string)
    
    # Convert binary data to a BytesIO object
    image_bytes = BytesIO(image_data)
    
    # Open the image with PIL
    pil_image = Image.open(image_bytes)
    
    return pil_image

@app.post("/generate")
async def generate_image(
    body: SDBody
):
    try:
        
        prompt = body.prompt
        depth = body.depth
        image = body.image
        print(prompt)
        
        depth = base64_to_pil(depth).convert("L")
        
        depth.save("depth.png")    
        
        with torch.no_grad():
            res = text2img_pipeline(
                prompt,
                negative_prompt='upper floors, roof, people, reflectionscartoon, painting, illustration, (worst quality, low quality, normal quality:2)',
                image = depth,
                height = depth.height,
                width = depth.width,
                controlnet_conditioning_scale=1.0,
                num_inference_steps=20,
                generator=generator
            ).images[0]
            
        res.save("generated_image_1.png")  
            
        # Generate image from the prompt
        image = base64_to_pil(image)

        image.save("input_image_1.png")
        
        image_np = np.array(image)
        image_alpha = image_np[:,:,3:]/255
        image_rgb = image_np[:,:,:3]
        blended = (image_alpha * image_rgb + (1 - image_alpha) * res).astype(np.uint8)
        canny = get_canny_image(blended)
        blended = Image.fromarray(blended)
        
        canny.save("canny.png")
        
        blended.save("input_image_2.png")
        
        with torch.no_grad():
            res = img2img_pipeline(
                prompt,
                negative_prompt='upper floors, roof, people, reflectionscartoon, painting, illustration, (worst quality, low quality, normal quality:2)',
                image = blended,
                control_image = canny,
                num_inference_steps=20,
                strength=0.2,
                height = canny.height,
                width = canny.width,
                controlnet_conditioning_scale=0.5,
                generator=generator
            ).images[0]
            
        res.save("generated_image_2.png")        

        # Save the generated image to a BytesIO object
        img_byte_arr = BytesIO()
        res.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        # Return the image as a StreamingResponse
        return {'res':img_b64}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
