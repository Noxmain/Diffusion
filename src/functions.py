from IPython.display import HTML, display, clear_output
import numpy as np
import torch
from PIL import Image
import base64
import io
import os

os.makedirs("../output", exist_ok=True)

def tensor_as_image(tensor):
    # convert an image tensor to an image object
    tensor = (tensor / 2 + 0.5).clamp(0, 1) # map from [-1,1] to [0,1]
    tensor = tensor.permute(0, 2, 3, 1).numpy()[0] # change order of tensor
    tensor = (tensor * 255).round().astype("uint8") # convert to image values
    return Image.fromarray(tensor) # return image object

def image_as_tensor(image):
    # convert an image object to an image tensor
    image = torch.from_numpy(np.array(image)) # create tensor
    image = image.permute(2, 0, 1).unsqueeze(0) # change order of tensor
    image = image / 255 * 2 - 1 # map from [0,255] to [-1,1]
    return image # return image tensor

def image_as_html(image):
    # convert an image object to an html tag
    buffer = io.BytesIO() # create buffer
    image.save(buffer, format="PNG") # fill buffer
    image = buffer.getvalue() # get bytes
    image = base64.b64encode(image).decode() # convert bytes
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

def show_images(*tensors):
    # show image tesors side by side
    show_table([[tensor_as_html(t) for t in tensors]])

def generate_noise(image_size):
    new_noise = torch.randn((1, 3, image_size, image_size))
    return new_noise

def user_input_patch():
    patch_size = None
    patch_position_x = None
    patch_position_y = None

    while patch_position_x is None or patch_position_x < 0 or patch_position_x > 256:
        try:
            patch_position_x = int(input("Enter the X position of the patch (0-256): "))
            if patch_position_x < 0 or patch_position_x > 256:
                print("Value out of range. Please enter a number between 0 and 256.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while patch_position_y is None or patch_position_y < 0 or patch_position_y > 256:
        try:
            patch_position_y = int(input("Enter the Y position of the patch (0-256): "))
            if patch_position_y < 0 or patch_position_y > 256:
                print("Value out of range. Please enter a number between 0 and 256.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    max_patch_size = min(256 - patch_position_x, 256 - patch_position_y)
    while patch_size is None or patch_size > max_patch_size  or patch_size < 0:
        try:
            patch_size = int(input(f"Enter patch size (max {max_patch_size}): "))
            if patch_size > max_patch_size or patch_size < 0:
                print(f"Value out of range. Please enter a number between 0 and {max_patch_size}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    return patch_size, patch_position_x, patch_position_y


def user_input_noise_scaling_factor():
    noise_scaling_factor = None
    while noise_scaling_factor is None or not (0.0 <= noise_scaling_factor <= 1.0):
        try:
            noise_scaling_factor = float(input("Enter noise scaling factor (0.0-1.0): "))
            if not (0.0 <= noise_scaling_factor <= 1.0):
                print("Value out of range. Please enter a number between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a float.")

    return noise_scaling_factor