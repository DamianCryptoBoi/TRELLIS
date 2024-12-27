from fastapi import FastAPI, Form
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
# from openai import OpenAI
from time import time
import base64
from fastapi.responses import Response, StreamingResponse
import os

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--retry", type=int, default=3)
    return parser.parse_args()

args = get_args()

class Router():
    def __init__(self):
        self.base_url = 'http://localhost'
        self.generation_ports = [8093 +  i*100 for i in range(args.gpus)]
        # self.validation_ports = [8094 +  i*100 for i in range(args.gpus)]
        self.next_index = 0
        print("Router initialized")
        print("Generation ports:", self.generation_ports)
    def get_endpoint(self, test=False):
        if test:
            print("get test endpoint")
            endpoint = f'{self.base_url}:{self.validation_ports[self.next_index]}/test'
        endpoint = f'{self.base_url}:{self.generation_ports[self.next_index]}/generate/'
        self.next_index = (self.next_index + 1) % len(self.generation_ports)
        return endpoint
    
router = Router()

app = FastAPI()
@app.post("/generate/")
async def generate(prompt: str = Form(), validation_threshold: float = 0.6):
    print(f"prompt: " + prompt)
    print(f"validation_threshold: " + str(validation_threshold))

    data = {
        "prompt": prompt,
        "validation_threshold": validation_threshold
    }
    
    try:
        count = 0
        while count < args.retry:
            count += 1
            async with httpx.AsyncClient(follow_redirects=True) as client:
                endpoint = router.get_endpoint()
                print(f"Requesting from {endpoint}")
                response = await client.post(endpoint, data=data, timeout=100)
            ply_path = response.json().get("ply_path")
            if len(ply_path)>0 and os.path.exists(ply_path):
                with open(ply_path, "rb") as f:
                    buffer = f.read()
                buffer = base64.b64encode(buffer).decode("utf-8")
                os.remove(ply_path)
                return Response(buffer, media_type="application/octet-stream")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""
    
@app.post("/test")
async def test(prompt: str = Form(), validation_threshold: float = 0.6):
    print(f"prompt: " + prompt)
    print(f"validation_threshold: " + str(validation_threshold))

    data = {
        "prompt": prompt,
        "validation_threshold": validation_threshold
    }
    
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            endpoint = router.get_endpoint(test=True)
            print(f"Requesting from {endpoint}")
            response = await client.post(endpoint, data=data, timeout=100)
            return  response.json().get("score",0)
    except Exception as e:
        print(f"Error: {e}")
        return ""
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8999)