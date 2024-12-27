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
        endpoint = f'{self.base_url}:{self.generation_ports[self.next_index]}/'
        if test:
            endpoint+= 'test'
        else:
            endpoint += 'generate/'
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
        ply_path_list = []
        best_ply_path = ""
        best_score = 0
        while count < args.retry:
            count += 1
            async with httpx.AsyncClient(follow_redirects=True) as client:
                endpoint = router.get_endpoint()
                print(f"Requesting from {endpoint}")
                response = await client.post(endpoint, data=data, timeout=100)
                data = response.json()
                print(data)
            ply_path = data.get("ply_path","")
            score = data.get("score",0)
            ply_path_list.append(ply_path)
            if score >= validation_threshold and os.path.exists(ply_path):
                if score > best_score:
                    best_score = score
                    best_ply_path = ply_path
                    print(f"Best score: {best_score}")
                    if score >=0.8:
                        print("Found a good ply. Stopping the loop.")
                        break
        with open(best_ply_path, "rb") as f:
            buffer = f.read()
        buffer = base64.b64encode(buffer).decode("utf-8")
        for ply_path in ply_path_list:
            os.remove(ply_path)
        return Response(buffer, media_type="application/octet-stream")
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