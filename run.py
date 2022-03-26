import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss


"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContefntLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    conv_counter = 0

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    # normalization = TODO
    normalization = Normalization()
    model = nn.Sequential(normalization)

    for layer in cnn:
        if type(layer) == nn.Conv2d:
            conv_counter += 1

        if f'conv_{conv_counter}' in content_layers:
            content_out = model(content_img).detach()
            content_loss = ContentLoss(content_out)
            content_losses.append(content_loss)
            model.add_module(f'content_{conv_counter}', content_loss)

        if f'conv_{conv_counter}' in style_layers:
            style_out = model(style_img).detach()
            style_loss = StyleLoss(style_out)
            style_losses.append(style_loss)
            model.add_module(f'style_{conv_counter}', style_loss)

    # raise NotImplementedError()

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    # get the optimizer
    optimizer = get_image_optimizer(input_img)

    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    # def closure():
    # here
    # which does the following:
    # clear the gradients
    # compute the loss and it's gradient
    # return the loss

    iter_num = 0



    def add_loss(losses):
        loss_sum = .0
        for loss in losses:
            loss_sum += loss.loss
        return loss_sum

    def content_closure():
        optimizer.zero_grad()
        model(input_img)
        sum_content_loss = add_loss(content_losses) * content_weight
        sum_content_loss.backward()

        nonlocal iter_num
        iter_num += 1
        if iter_num % 50 == 0:
            print(f"iter at {iter_num},  content loss is {sum_content_loss:.6f}")

        return sum_content_loss

    def style_closure():
        optimizer.zero_grad()
        model(input_img)
        sum_content_loss = add_loss(content_losses) * content_weight
        sum_style_loss = add_loss(style_losses) * style_weight
        sum_loss = sum_content_loss + sum_style_loss
        sum_loss.backward()

        nonlocal iter_num
        iter_num += 1
        if iter_num % 50 == 0:
            print(f"iter at {iter_num}, style loss is {sum_content_loss:.6f}, content loss is {sum_style_loss:.6f}")


        return sum_content_loss + sum_style_loss


    while iter_num < num_steps:
        if use_content and (not use_style):
            optimizer.step(content_closure)
        else:
            optimizer.step(style_closure)

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step

    with torch.no_grad():
        torch.clamp(input_img, min=0, max=1)
    # raise NotImplementedError()

    # make sure to clamp once you are done

    return input_img


def main(style_img_path, content_img_path):
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)

    # interative MPL
    plt.ion()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    print(content_img.size(), "size")
    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    input_img = torch.randn(content_img.size()).to(device)# random noise of the size of content_img on the correct device
    output = run_optimization(cnn, content_img, style_img, input_img,  use_style=False, use_content=True)# reconstruct the image from the noise

    plt.figure()
    imshow(output, title='Reconstructed Image')

    # texture synthesis
    # print("Performing Texture Synthesis from white noise initialization")
    # input_img = torch.randn(content_img.size()).to(device) # random noise of the size of content_img on the correct device
    # output = run_optimization(cnn, content_img, style_img, input_img,  use_style=True, use_content=True, num_steps=1000) #synthesize a texture like style_image
    #
    # plt.figure()
    # imshow(output, title='Synthesized Texture')

    # style transfer
    # input_img = random noise of the size of content_img on the correct device
    # output = transfer the style from the style_img to the content image

    # plt.figure()
    # imshow(output, title='Output Image from noise')

    print("Performing Style Transfer from content image initialization")
    input_img = content_img.clone()
    output = run_optimization(cnn, content_img, style_img, input_img,  use_style=True, use_content=True, num_steps=1000)# transfer the style from the style_img to the content image

    plt.figure()
    imshow(output, title='Output Image from content img')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:3]
    main(*args)
