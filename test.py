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

# 624701db8f84649fca7eb5ad56ed82aaa07a852d1e904ef5d2d9ef1da1434915