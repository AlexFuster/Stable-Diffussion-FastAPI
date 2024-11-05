from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import matplotlib.pyplot as plt

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

pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
    model_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
).to('cuda')

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

print(pipe.config)

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...), 
    image: UploadFile = File(...)
):
    try:
        print(prompt)
        # Generate image from the prompt
        image_bytes = await image.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        image_np = np.array(image)
        control_image = make_control_image(image_np)

        noise = np.random.uniform(0,255,image_np.shape).astype(np.uint8)
        mask_bg = cv2.inRange(image_np, np.array([0,0,0]), np.array([0,0,0]))
        mask_bg=np.stack([mask_bg,mask_bg,mask_bg],axis=2)

        floor_value = np.array([183, 140, 100])
        all_floor = (np.zeros(image_np.shape)+floor_value.reshape((1,1,-1))).astype(np.uint8)
        mask_floor = cv2.inRange(image_np, floor_value, floor_value)
        mask_floor=np.stack([mask_floor,mask_floor,mask_floor],axis=2)
        
        input_image_1 = np.where(mask_bg == 0, image_np, noise)

        input_image_1 = Image.fromarray(input_image_1)

        input_image_1.save("input_image_1.png")
        control_image.save("control_image.png")

        #plt.imshow(mask_bg)
        #plt.show()

        res = pipe(
            'a realistic grassy town park',
            negative_prompt='objects, structures, people, playground, reflectionscartoon, painting, illustration, (worst quality, low quality, normal quality:2)',
            image = input_image_1,
            control_image = control_image,
            mask_image= Image.fromarray(mask_bg),
            num_inference_steps=25,
            strength=1.0,
            height = image.height,
            width = image.width,
            controlnet_conditioning_scale=0.0,
            padding_mask_crop=32
        ).images[0]

        res.save("generated_image_1.png")

        
        #plt.imshow(mask_floor)
        #plt.show()

        input_image_2 = res
        input_image_2.save("input_image_2.png")

        res = pipe(
            'a realistic dirt square',
            negative_prompt='objects, sand, reflectionscartoon, painting, illustration, (worst quality, low quality, normal quality:2)',
            image = input_image_2,
            control_image = control_image,
            mask_image= Image.fromarray(mask_floor),
            num_inference_steps=20,
            strength=1.0,
            height = image.height,
            width = image.width,
            controlnet_conditioning_scale=0.8,
            padding_mask_crop=32
        ).images[0]

        res.save("generated_image_2.png")

        input_image_3 = res
        input_image_3.save("input_image_3.png")
        mask_play = 255-np.minimum(mask_bg+mask_floor,255)

        #plt.imshow(mask_play)
        #plt.show()

        res = pipe(
            'a sharp realistic beautiful playground in a dirt square in a town park',
            negative_prompt='blurry edges, color changes, unexpected objects, reflectionscartoon, painting, illustration, (worst quality, low quality, normal quality:2)',
            image = input_image_3,
            control_image = control_image,
            mask_image= Image.fromarray(255-mask_bg),
            num_inference_steps=25,
            strength=0.5,
            height = image.height,
            width = image.width,
            controlnet_conditioning_scale=2.0,
            padding_mask_crop=32
        ).images[0]

        res.save("generated_image_3.png")


        #'a realistic beautiful playground in a dirt square in a town park'
        


        # Save the generated image to a BytesIO object
        img_byte_arr = BytesIO()
        res.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Return the image as a StreamingResponse
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
