import torch
import diffusers
from PIL import Image
from tqdm import tqdm
import numpy as np

def as_image(tensor):
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    tensor = tensor.permute(0, 2, 3, 1).numpy()[0]
    tensor = (tensor * 255).round().astype("uint8")
    return Image.fromarray(tensor)

# load input image
image = Image.open("output/ddpm_9.png")
image = torch.from_numpy(np.array(image))
image = image.permute(2, 0, 1).unsqueeze(0)
image = torch.tensor(image)
image = image / 255.0 * 2 - 1
width = image.shape[2]
height = image.shape[3]

# setup model
model_id = "google/ddpm-celebahq-256"
model = diffusers.UNet2DModel.from_pretrained(model_id)
scheduler = diffusers.DDPMScheduler.from_pretrained(model_id)
scheduler.set_timesteps(50)

# setup output image
noise = torch.randn((1, 3, width, height))
output = Image.new("RGB", (width * 50, height * 50))
output.paste(as_image(image), (0, 0))

# output generation
for i in tqdm(reversed(range(50))):
    current = scheduler.add_noise(image, noise, scheduler.timesteps[i])
    output.paste(as_image(current), (0, height * (50 - i)))
    for j, t in enumerate(scheduler.timesteps[i:]):
        with torch.no_grad():
            predicted_noise = model(current, t).sample
            current = scheduler.step(predicted_noise, t, current).prev_sample
            output.paste(as_image(current), (width * (j + 1), height * (50 - i)))

output.save("output/process.png")
