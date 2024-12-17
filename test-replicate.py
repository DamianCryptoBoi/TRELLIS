import gradio as gr
import replicate
from PIL import Image
import requests
from io import BytesIO

def generate_image(prompt: str):
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
    # return image
    output = replicate.run(
    "recraft-ai/recraft-20b",
    input={
        "size": "1024x1024",
        "style": "digital_illustration/3d",
        "prompt": prompt
        }
    )
    image_url = output
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image
    

iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="3D Model Image Generator",
    description="Enter a prompt to generate a 3D model image."
)

iface.launch()