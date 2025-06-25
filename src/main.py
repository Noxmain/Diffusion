import torch
import diffusers
from PIL import Image
from tqdm import tqdm

# setup
model_id = "google/ddpm-celebahq-256" # select pretrained model
# pipe = diffusers.DDIMPipeline.from_pretrained(model_id)
model = diffusers.UNet2DModel.from_pretrained(model_id)
scheduler = diffusers.DDPMScheduler.from_pretrained(model_id)
scheduler.set_timesteps(50) # set timesteps

# input prepraration
image_size = model.config.sample_size # get image size
input = torch.randn((1, 3, image_size, image_size)) # sample random noise

# output generation
current = input
for t in tqdm(scheduler.timesteps):
    with torch.no_grad(): # no training
        predicted_noise = model(current, t).sample # inference
        current = scheduler.step(predicted_noise, t, current).prev_sample # denoise

# output
# output = pipe().images[0]
output = (current / 2 + 0.5).clamp(0, 1) # map from [-1,1] to [0,1]
output = output.permute(0, 2, 3, 1).numpy()[0] # change order of tensor
output = (output * 255).round().astype("uint8") # convert to image values
output = Image.fromarray(output) # convert to image
output.save("output/output.png") # save image
