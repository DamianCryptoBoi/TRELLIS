import gradio as gr
from together import Together
import base64
from io import BytesIO
from PIL import Image

client = Together()

def generate_image(prompt: str):
    imageCompletion = client.images.generate(
        model="black-forest-labs/FLUX.1-schnell-Free",
        width=1024,
        height=1024,
        steps=4,
        prompt=prompt,
        response_format="b64_json"
    )
    b64_json = imageCompletion.data[0].b64_json
    image_data = base64.b64decode(b64_json)
    image = Image.open(BytesIO(image_data))
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="3D Model Image Generator",
    description="Enter a prompt to generate a 3D model image."
)

iface.launch()