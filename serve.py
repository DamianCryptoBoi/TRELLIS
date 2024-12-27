

import os
from typing import *
import numpy as np
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
from fastapi import FastAPI, UploadFile, File,Form
from fastapi.responses import JSONResponse
import uvicorn
from together import Together
import time
import base64
from io import BytesIO
import requests
from pydantic import BaseModel
from fastapi.responses import Response, StreamingResponse

import torch
# from diffusers import HunyuanDiTPipeline

# from image_gen import Text2Image
from sharpen_img import laplacian_filter, unsharp_mask
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--v_port", type=int, default=8094)
    return parser.parse_args()


args = get_args()

client = Together()

MAX_SEED = np.iinfo(np.int32).max
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
app = FastAPI()
# img_generator = Text2Image()
os.makedirs("/gen-data", exist_ok=True)

def generate_image(prompt: str):
    start_time = time.time()
    prompt = f"{prompt}, white background, 3D style, high quality"
    # prompt = f"highly detailed and colorful 3d model of a {prompt}, white background"
    image = client.images.generate(
        model="black-forest-labs/FLUX.1-schnell",
        width=1024,
        height=1024,
        steps=4,
        prompt=prompt,
        response_format="b64_json"
    )
    end_time = time.time()
    
    print("Prompt:", prompt)
    print("Time taken to generate image:", end_time - start_time)
    return image.data[0].b64_json

    # output = replicate.run(
    #     "black-forest-labs/flux-dev",
    #     input={
    #         "prompt": prompt,
    #         "go_fast": True,
    #         "guidance": 3.5,
    #         "megapixels": "1",
    #         "num_outputs": 1,
    #         "aspect_ratio": "1:1",
    #         "output_format": "webp",
    #         "output_quality": 80,
    #         "prompt_strength": 0.8,
    #         "num_inference_steps": 28
    #     }
    # )
    # image_url = output[0]
    # response = requests.get(image_url)
    # image = Image.open(BytesIO(response.content))
    # end_time = time.time()
    # print("Time taken to generate image:", end_time - start_time)
    # return image

    # output = replicate.run(
    # "recraft-ai/recraft-20b",
    # input={
    #     "size": "1024x1024",
    #     "style": "digital_illustration/3d",
    #     "prompt": prompt
    #     }
    # )
    # image_url = output
    # response = requests.get(image_url)
    # image = Image.open(BytesIO(response.content))
    # image = img_generator(prompt)
    # end_time = time.time()
    # print("Time taken to generate image:", end_time - start_time)
    # return image

def pack_state(gs: Gaussian) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        }
    }


def image_to_3d_test(prompt: str, image: Image.Image, ss_guidance_strength: float = 7.5, ss_sampling_steps: int = 12, slat_guidance_strength: float = 3, slat_sampling_steps: int = 12) -> Tuple[dict, str]:
    try:
        start_time = time.time()
        seed = np.random.randint(0, MAX_SEED)
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        ply_path = f"./gen-data/{seed}.ply"
        outputs['gaussian'][0].save_ply(ply_path)
        print("Ply file saved at:", ply_path)
        #read the ply file
        with open(ply_path, "rb") as f:
            buffer = f.read()
        buffer = base64.b64encode(buffer).decode("utf-8")
        response = requests.post("http://localhost:8094/validate_ply/", json={"prompt": prompt, "data": buffer})
        end_time = time.time()
        score = response.json().get("score", 0)
        print(response.json())
        print("Time taken to convert image to 3D:", end_time - start_time)
        # remove the ply file
        os.remove(ply_path)
        return score
    except Exception as e:
        print(f"Error: {e}")
        return 0

def image_to_3d(prompt: str, validation_threshold: int = 0.6, ss_guidance_strength: float = 7.5, ss_sampling_steps: int = 12, slat_guidance_strength: float = 3, slat_sampling_steps: int = 12) -> Tuple[dict, str]:
    start_time = time.time()
    count = 0
    try:
        while count < 1:

            b64_json = generate_image(prompt)
            image_data = base64.b64decode(b64_json)
            image = Image.open(BytesIO(image_data))
            seed = np.random.randint(0, MAX_SEED)
            outputs = pipeline.run(
                image,
                seed=seed,
                formats=["gaussian"],
                preprocess_image=True,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )
            ply_path = f"./gen-data/{seed}.ply"
            outputs['gaussian'][0].save_ply(ply_path)
            print("Ply file saved at:", ply_path)
            #read the ply file
            with open(ply_path, "rb") as f:
                buffer = f.read()
            buffer = base64.b64encode(buffer).decode("utf-8")
            # return buffer
            response = requests.post(f"http://localhost:{args.v_port}/validate_ply/", json={"prompt": prompt, "data": buffer})
            end_time = time.time()
            score = response.json().get("score", 0)
            print("prompt:", prompt)
            print(response.json())
            print("Time taken to convert image to 3D:", end_time - start_time)
            # remove the ply file
            # os.remove(ply_path)
            if score >= validation_threshold:
                return ply_path, score
            else:
                os.remove(ply_path)
            count += 1
        return '', 0
    except Exception as e:
        print(f"Error: {e}")
        return '', 0

@app.post("/test")
async def test(prompt: str = Form()):
    b64_json = generate_image(prompt)
    image_data = base64.b64decode(b64_json)
    image = Image.open(BytesIO(image_data))
    # image = generate_image(prompt)
    score = image_to_3d_test(prompt, image)
    return JSONResponse(content={"score":score})


@app.post("/generate/")
async def generate(prompt: str = Form(), validation_threshold: float = 0.6):
    data = image_to_3d(prompt, validation_threshold)
    return JSONResponse(content={"ply_path":data[0], "score":data[1]})

# Launch the Gradio app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)