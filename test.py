import gradio as gr
from together import Together
import base64
from io import BytesIO
from PIL import Image
import time

# client = Together()
from openai import OpenAI
import time
client = OpenAI()

def generate_image(prompt: str):
    start = time.time()
    imageCompletion = client.images.generate(
        model="black-forest-labs/FLUX.1-schnell-Free",
        width=1024,
        height=1024,
        steps=4,
        prompt=prompt,
        response_format="b64_json",
        n=4
    )
    end = time.time()
    print(len(imageCompletion.data))
    print("Time taken to generate image:", end - start)
    # b64_json = imageCompletion.data[0].b64_json
    # image_data = base64.b64decode(b64_json)
    # image = Image.open(BytesIO(image_data))
    # return image
generate_image("3d model of a yellow bear holding a sword, white background")
# iface = gr.Interface(
#     fn=generate_image,
#     inputs="text",
#     outputs="image",
#     title="3D Model Image Generator",
#     description="Enter a prompt to generate a 3D model image."
# )

# iface.launch()