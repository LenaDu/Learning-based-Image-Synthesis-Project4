import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
import os
import time
import torch.optim as optim
import imageio
from torchvision.utils import save_image
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss, gram_matrix
from models import CycleGenerator, PatchDiscriminator
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop



"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_7']
style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_7', 'conv_9']


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


    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    # normalization = TODO
    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)

    layer_counter = 0

    for layer in cnn:
        layer_counter += 1
        if type(layer) == nn.ReLU:
            model.add_module(f'ReLU{layer_counter}', nn.ReLU(inplace=False))

        else:
            model.add_module(f'layer{layer_counter}', layer)

        if f'conv_{layer_counter}' in content_layers:
            content_out = model(content_img).detach()
            content_loss = ContentLoss(content_out)
            content_losses.append(content_loss)
            model.add_module(f'content_loss_{layer_counter}', content_loss)

        if f'conv_{layer_counter}' in style_layers:
            style_out = model(style_img).detach()
            style_loss = StyleLoss(style_out)
            style_losses.append(style_loss)
            model.add_module(f'style_loss_{layer_counter}', style_loss)

    cut_index = 0
    for i in range(len(model)):
        if type(model[i]) == ContentLoss or type(model[i]) == StyleLoss:
            cut_index = i

    model = model[:(cut_index+1)]

    # raise NotImplementedError()
    print(model)
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
        with torch.no_grad():
            input_img.clamp_(0, 1)
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
        with torch.no_grad():
            input_img.clamp_(0, 1)
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
        input_img.clamp_(0, 1)
    # make sure to clamp once you are done

    return input_img


def main(style_img_path, content_img_path, output_dir):
    # we've loaded the images for you
    style_weight = 1000000
    content_weight = 1

    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)
    style_img_name = style_img_path.split('/')[-1].split('.')[0]
    content_img_name = content_img_path.split('/')[-1].split('.')[0]
    comment = f'conv1+3+5+7+9_s{style_weight:06d}_c{content_weight:1d}'

    # interative MPL
    plt.ion()

    ## implement cropping
    content_width, content_height = F.get_image_size(content_img)
    style_width, style_height = F.get_image_size(style_img)

    width_ratio = style_width / content_width
    height_ratio = style_height / content_height
    style_ratio = style_height / style_width

    if width_ratio >= 1 and height_ratio >= 1:
        pass
    elif width_ratio < 1 and height_ratio >= 1:
        new_width = content_width
        new_height = int(new_width * style_ratio) + 1
        style_img = F.resize(style_img, (new_height, new_width))
    elif width_ratio >= 1 and height_ratio < 1:
        new_height = content_height
        new_width = int(new_height / style_ratio) + 1
        style_img = F.resize(style_img, (new_height, new_width))
    else:
        ratio = min(width_ratio, height_ratio)
        new_height = int(style_height / ratio) + 1
        new_width = int(style_width / ratio) + 1
        style_img = F.resize(style_img, (new_height, new_width))

    # crop
    # top, left, height, width = 0, 0, content_height, content_width
    top, left, height, width = RandomCrop.get_params(style_img, (content_height, content_width))
    style_img = F.crop(style_img, top=top, left=left, height=height, width=width)

    print(style_img.size())
    print(content_img.size())
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    ################################
    #    Style & Content Image     #
    ################################
    plt.figure()
    plt.axis('off')
    imshow(style_img, title='Style Image')
    # plt.savefig(f"{output_dir}/style_image_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    plt.figure()
    plt.axis('off')
    imshow(content_img, title='Content Image')
    # plt.savefig(f"{output_dir}/content_image_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    # generator = CycleGenerator()
    # generator = generator.features.to(device).eval()

    print(content_img.size(), "size")
    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")


    ################################
    #        Reconstruction        #
    ################################

    ###################### Noise 1 #######################
    input_img = torch.randn(content_img.size()).to(device)# random noise of the size of content_img on the correct device

    plt.figure()
    plt.axis('off')
    imshow(input_img, title='noise1')
    # plt.savefig(f"{output_dir}/noise1_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    output = run_optimization(cnn, content_img, style_img, input_img,  use_style=False, use_content=True)# reconstruct the image from the noise


    plt.figure()
    plt.axis('off')
    imshow(output, title='Reconstructed Image(noise1)')
    plt.savefig(f"{output_dir}/reconstructed_image(noise1)_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')


    ####################### Noise 2 #######################
    input_img = torch.randn(content_img.size()).to(device)# random noise of the size of content_img on the correct device

    plt.figure()
    plt.axis('off')
    imshow(input_img, title='noise2')
    plt.savefig(f"{output_dir}/noise2_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    output = run_optimization(cnn, content_img, style_img, input_img,  use_style=False, use_content=True)# reconstruct the image from the noise

    plt.figure()
    plt.axis('off')
    imshow(output, title='Reconstructed Image(noise2)')
    plt.savefig(f"{output_dir}/reconstructed_image(noise2)_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    # ####################### Noise #######################
    # input_img = torch.randn(content_img.size()).to(device)# random noise of the size of content_img on the correct device
    # output = run_optimization(cnn, content_img, style_img, input_img,  use_style=False, use_content=True)# reconstruct the image from the noise
    #
    # plt.figure()
    # plt.axis('off')
    # imshow(output, title='Reconstructed Image')
    # plt.savefig(f"{output_dir}/reconstructed_image_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    ################################
    #       Texture Synthesis      #
    ################################
    # texture

    print("Performing Texture Synthesis from white noise initialization")

    ######################## Noise 1 #######################
    input_img = torch.randn(content_img.size()).to(device) # random noise of the size of content_img on the correct device

    plt.figure()
    plt.axis('off')
    imshow(input_img, title='noise1')
    plt.savefig(f"{output_dir}/noise1_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    output = run_optimization(cnn, input_img, style_img, input_img,  use_style=True, use_content=True) #synthesize a texture like style_image

    plt.figure()
    plt.axis('off')
    imshow(output, title='Synthesized Texture(noise1)')
    plt.savefig(f"{output_dir}/texture_image(noise1)_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    ######################## Noise 2 #######################

    input_img = torch.randn(content_img.size()).to(device)  # random noise of the size of content_img on the correct device

    plt.figure()
    plt.axis('off')
    imshow(input_img, title='noise2')
    plt.savefig(f"{output_dir}/noise2_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    output = run_optimization(cnn, input_img, style_img, input_img, use_style=True,
                              use_content=True)  # synthesize a texture like style_image

    plt.figure()
    plt.axis('off')
    imshow(output, title='Synthesized Texture(noise2)')
    plt.savefig(f"{output_dir}/texture_image(noise2)_{style_img_name}_{content_img_name}_{comment}.png",
                bbox_inches='tight')

    print("Performing Texture Synthesis from white noise initialization")
    input_img = torch.randn(content_img.size()).to(device) # random noise of the size of content_img on the correct device
    output = run_optimization(cnn, input_img, style_img, input_img,  use_style=True, use_content=True) #synthesize a texture like style_image

    plt.figure()
    plt.axis('off')
    imshow(output, title='Synthesized Texture')
    plt.savefig(f"{output_dir}/texture_image_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    ################################
    #    Style Transfer (noise)    #
    ################################
    # style transfer
    input_img = torch.randn(content_img.size()).to(device) # noise of the size of content_img on the correct device
    start_time = time.time()
    output = run_optimization(cnn, content_img, style_img, input_img,  use_style=True, use_content=True, style_weight=style_weight, content_weight=content_weight) #transfer the style from the style_img to the content image
    noise_time = time.time() - start_time

    plt.figure()
    plt.axis('off')
    # plt.figtext(0.25, 0, f'style_weight: {style_weight}, content_weight:{content_weight}')

    plt.figtext(0.25, 0, f'time: {noise_time}')

    imshow(output, title='Output Image from noise')
    plt.savefig(f"{output_dir}/output_from_noise_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')


    ################################
    #    Style Transfer (image)    #
    ################################
    print("Performing Style Transfer from content image initialization")
    input_img = content_img.clone()
    start_time = time.time()
    output = run_optimization(cnn, content_img, style_img, input_img,  use_style=True, use_content=True, style_weight=style_weight,
                              content_weight=content_weight, num_steps=500)# transfer the style from the style_img to the content image
    image_time = time.time() - start_time

    plt.figure()
    plt.axis('off')
    # plt.figtext(0.25, 0, f'style_weight: {style_weight}, content_weight:{content_weight}')

    # plt.figtext(0.25, 0, f'time: {image_time}')

    imshow(output, title='Output Image from content img(Original)')
    plt.savefig(f"{output_dir}/output_from_content_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')

    ################################
    # Style Transfer (feedforward) #
    ################################
    # print("Performing Style Transfer from content image initialization")
    # input_img = content_img.clone()
    # start_time = time.time()
    # output = run_optimization_feedforward(generator, content_img, style_img, input_img, num_steps=1200)# transfer the style from the style_img to the content image
    # image_time = time.time() - start_time

    plt.figure()
    plt.axis('off')
    plt.figtext(0.25, 0, f'style_weight: {style_weight}, content_weight:{content_weight}')
    plt.figtext(0.25, 0, f'time: {image_time}')

    imshow(output, title='Output Image from content img(Feedforward)')
    plt.savefig(f"{output_dir}/output_from_content_{style_img_name}_{content_img_name}_{comment}.png", bbox_inches='tight')


    plt.ioff()
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:3]
    output_dir = 'output'
    main(*args, output_dir=output_dir)
    try:
        os.mkdir(output_dir)
    except:
        pass
