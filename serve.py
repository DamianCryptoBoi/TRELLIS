

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
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8093)
    return parser.parse_args()


args = get_args()

client = Together()

MAX_SEED = np.iinfo(np.int32).max

app = FastAPI()
os.makedirs("/tmp", exist_ok=True)

def generate_image(prompt: str):
    start_time = time.time()
    prompt = f"highly detailed and colorful 3d model of a {prompt}, white background"
    image = client.images.generate(
        model="black-forest-labs/FLUX.1-schnell-Free",
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

def image_to_3d(prompt: str, image: Image.Image, validation_threshold: int = 0.68, ss_guidance_strength: float = 7.5, ss_sampling_steps: int = 12, slat_guidance_strength: float = 3, slat_sampling_steps: int = 12) -> Tuple[dict, str]:
    start_time = time.time()
    count = 0

    while count < 1:
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
        os.remove(ply_path)
        return buffer
        # response = requests.post("http://localhost:8094/validate_ply/", json={"prompt": prompt, "data": buffer})
        # end_time = time.time()
        # score = response.json().get("score", 0)
        # print("prompt:", prompt)
        # print(response.json())
        # print("Time taken to convert image to 3D:", end_time - start_time)
        # # remove the ply file
        # os.remove(ply_path)
        # if score >= 0.5:
        #     return buffer
        # count += 1
    return ''

@app.post("/test")
async def test(prompt: str = Form()):
    b64_json = generate_image(prompt)
    image_data = base64.b64decode(b64_json)
    image = Image.open(BytesIO(image_data))
    state = image_to_3d_test(prompt, image)
    return JSONResponse(content=state)


@app.post("/generate/")
async def generate(prompt: str = Form(), validation_threshold: float = 0.68):
    b64_json = generate_image(prompt)
    image_data = base64.b64decode(b64_json)
    image = Image.open(BytesIO(image_data))
    buffer = image_to_3d(prompt, image, validation_threshold)
    return Response(buffer, media_type="application/octet-stream")

# Launch the Gradio app
if __name__ == "__main__":
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    uvicorn.run(app, host="0.0.0.0", port=args.port)