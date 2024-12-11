from together import Together

client = Together()

imageCompletion = client.images.generate(
    model="black-forest-labs/FLUX.1-schnell-Free",
    width=1024,
    height=1024,
    steps=4,
    prompt="3d model of a red bicycle, white background",
)

print(imageCompletion.data[0].url)