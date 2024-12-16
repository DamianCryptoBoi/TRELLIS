from openai import OpenAI
import time
client = OpenAI()

start = time.time()
a = client.images.generate(
  model="dall-e-3",
  prompt="3d model of a yellow bear holding a sword, white background",
  n=1,
  size="1024x1024",
  response_format="b64_json"
)

end = time.time()

print(a.data[0])
print("Time taken to generate image:", end - start)