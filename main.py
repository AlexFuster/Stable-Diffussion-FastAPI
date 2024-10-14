from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

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

def make_control_image(image):
    control_image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    control_image = cv2.Canny(control_image, low_threshold, high_threshold)
    control_image = control_image[:, :, None]
    control_image = np.concatenate([control_image, control_image, control_image], axis=2)
    control_image = Image.fromarray(control_image)
    return control_image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

model_path = '/home/alex/stable-diffusion-webui/models/Stable-diffusion/photon_v1.safetensors'

pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
    model_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
).to('cuda')

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


@app.post("/generate")
async def generate_image(
    prompt: str = Form(...), 
    image: UploadFile = File(...)
):
    try:
        # Generate image from the prompt
        image_bytes = await image.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        control_image = make_control_image(image)

        image.save("input_image.png")
        control_image.save("control_image.png")

        res = pipe(
            prompt,
            image = image,
            control_image = control_image,
            num_inference_steps=20,
            
        ).images[0]

        res.save("generated_image.png")

        # Save the generated image to a BytesIO object
        img_byte_arr = BytesIO()
        res.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Return the image as a StreamingResponse
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
