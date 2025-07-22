from IPython.display import HTML, display, clear_output
from io import BytesIO
from base64 import b64encode
from PIL import Image
from os import makedirs
from yaml import safe_load
from numpy import array
import torch

makedirs("../output", exist_ok=True)

def tensor_as_image(tensor):
    # convert an image tensor to an image object
    tensor = (tensor / 2 + 0.5).clamp(0, 1) # map from [-1,1] to [0,1]
    tensor = tensor.permute(0, 2, 3, 1).numpy()[0] # change order of tensor
    tensor = (tensor * 255).round().astype("uint8") # convert to image values
    return Image.fromarray(tensor) # return image object

def image_as_tensor(image):
    # convert an image object to an image tensor
    image = torch.from_numpy(array(image)) # create tensor
    image = image.permute(2, 0, 1).unsqueeze(0) # change order of tensor
    image = image / 255 * 2 - 1 # map from [0,255] to [-1,1]
    return image # return image tensor

def image_as_html(image):
    # convert an image object to an html tag
    buffer = BytesIO() # create buffer
    image.save(buffer, format="PNG") # fill buffer
    image = buffer.getvalue() # get bytes
    image = b64encode(image).decode() # convert bytes
    return f"<img src=\"data:image/png;base64,{image}\" />" # return html tag

def tensor_as_html(tensor):
    # convert an image tensor to an html tag
    return image_as_html(tensor_as_image(tensor))

def show(object):
    # clear the notebook cell and display an object
    clear_output(wait=True)
    display(object)

def show_table(table):
    # show a 2d nested list as table
    a = "<table><tr><td><center>"
    b = "</center></td></tr><tr><td><center>"
    c = "</center></td><td><center>"
    d = "</center></td></tr></table>"
    show(HTML(a + b.join(c.join(t) for t in table) + d))

def show_images(*tensors, labels=None):
    # show image tesors side by side
    if labels:
        show_table([[tensor_as_html(t) for t in tensors], labels])
    else:
        show_table([[tensor_as_html(t) for t in tensors]])

def generate_noise(image_size):
    # sample a random noise image tensor
    new_noise = torch.randn((1, 3, image_size, image_size))
    return new_noise

def config(key):
    # load an attribute from the config file
    with open("config.yaml") as f:
        return safe_load(f).get(key)
