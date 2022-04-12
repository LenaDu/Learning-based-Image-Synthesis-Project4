#  Project4 - Neural Style Transfer

###### Lena Du



#### B & W are explained in part4

* Stylize image from the previous homework. 
* *Implemented my own cropping method.*
* Tried to use a feedforward network to output style transfer results directly.



### Part 1 Content Reconstruction



#### 1.1 Content loss optimization

By using content loss to different layers, we can see the reconstructed image varies. When the applied layer is deeper, the detail of the reconstructed image has more noise. 

| Layer optimized content loss | Reconstructed Image                                          | Detail                                     |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| conv1                        | ![](output\content_loss\reconstructed_image_frida_kahlo_frida_kahlo_conv1.png) | ![](output/content_loss/detail_conv1.jpg)  |
| ***conv4***                  | ![](output\content_loss\reconstructed_image_frida_kahlo_frida_kahlo_conv4.png) | ![](output/content_loss/detail_conv4.jpg)  |
| conv7                        | ![](output\content_loss\reconstructed_image_frida_kahlo_frida_kahlo_conv7.png) | ![](output/content_loss/detail_conv7.jpg)  |
| conv10                       | ![](output\content_loss\reconstructed_image_frida_kahlo_frida_kahlo_conv10.png) | ![](output/content_loss/detail_conv10.jpg) |
| conv13                       | ![](output\content_loss\reconstructed_image_frida_kahlo_frida_kahlo_conv13.png) | ![](output/content_loss/detail_conv13.jpg) |



#### 1.2 Reconstruction from random noises

| Layer  | Noise1                                                       | Noise2                                                       | Reconstruction from noise1                                   | Reconstruction from noise2                                   |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| conv1  | ![](output\noise_content_loss\noise1_frida_kahlo_frida_kahlo_conv1.png) | ![](output\noise_content_loss\noise2_frida_kahlo_frida_kahlo_conv1.png) | ![](output\noise_content_loss\reconstructed_image(noise1)_frida_kahlo_frida_kahlo_conv1.png) | ![](output\noise_content_loss\reconstructed_image(noise2)_frida_kahlo_frida_kahlo_conv1.png) |
| conv4  | ![](output\noise_content_loss\noise1_frida_kahlo_frida_kahlo_conv4.png) | ![](output\noise_content_loss\noise2_frida_kahlo_frida_kahlo_conv4.png) | ![](output\noise_content_loss\reconstructed_image(noise1)_frida_kahlo_frida_kahlo_conv4.png) | ![](output\noise_content_loss\reconstructed_image(noise2)_frida_kahlo_frida_kahlo_conv4.png) |
| conv7  | ![](output\noise_content_loss\noise1_frida_kahlo_frida_kahlo_conv7.png) | ![](output\noise_content_loss\noise2_frida_kahlo_frida_kahlo_conv7.png) | ![](output\noise_content_loss\reconstructed_image(noise1)_frida_kahlo_frida_kahlo_conv7.png) | ![](output\noise_content_loss\reconstructed_image(noise2)_frida_kahlo_frida_kahlo_conv7.png) |
| conv10 | ![](output\noise_content_loss\noise1_frida_kahlo_frida_kahlo_conv10.png) | ![](output\noise_content_loss\noise2_frida_kahlo_frida_kahlo_conv10.png) | ![](output\noise_content_loss\reconstructed_image(noise1)_frida_kahlo_frida_kahlo_conv10.png) | ![](output\noise_content_loss\reconstructed_image(noise2)_frida_kahlo_frida_kahlo_conv10.png) |
| conv13 | ![](output\noise_content_loss\noise1_frida_kahlo_frida_kahlo_conv13.png) | ![](output\noise_content_loss\noise2_frida_kahlo_frida_kahlo_conv13.png) | ![](output\noise_content_loss\reconstructed_image(noise1)_frida_kahlo_frida_kahlo_conv13.png) | ![](output\noise_content_loss\reconstructed_image(noise2)_frida_kahlo_frida_kahlo_conv13.png) |



### Part 2 Texture Synthesis



#### 2.1 Style loss optimization

As we can see, when the content loss is optimized at deeper layers, the synthesized texture is more abstract and has more similar shapes but less similar colors. From my point of view, the combination of `conv1`, `conv3`, `conv5`, `conv7`, and `conv9` looks the best. Therefore, I am going to choose this configuration for my style loss.

| Layer optimized content loss           | Synthesized texture                                          |
| -------------------------------------- | ------------------------------------------------------------ |
| conv1, conv2, conv3, conv4, conv5      | ![](output\style_loss\texture_image_frida_kahlo_frida_kahlo_conv1-5.png) |
| *conv1, conv3, conv5, conv7, conv9*    | ![](output\style_loss\texture_image_frida_kahlo_frida_kahlo_conv1_3_5_7_9.png) |
| conv6, conv7, conv8, conv9, conv10     | ![](output\style_loss\texture_image_frida_kahlo_frida_kahlo_conv6-10.png) |
| conv11, conv12, conv13, conv14, conv15 | ![](output\style_loss\texture_image_frida_kahlo_frida_kahlo_conv11-15.png) |



#### 2.2 Reconstruction from random noises

| Layer                                  | Noise1                                                       | Noise2                                                       | Synthesized texture from noise1                              | Synthesized texturefrom noise2                               |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| conv1, conv2, conv3, conv4, conv5      | ![](output\noise_style_loss\noise1_starry_night_starry_night_conv1-5.png) | ![](output\noise_style_loss\noise2_starry_night_starry_night_conv1-5.png) | ![](output\noise_style_loss\texture_image(noise1)_starry_night_starry_night_conv1-5.png) | ![](output\noise_style_loss\texture_image(noise2)_starry_night_starry_night_conv1-5.png) |
| conv1, conv3, conv5, conv7, conv9      | ![](output\noise_style_loss\noise1_starry_night_starry_night_conv1+3+5+7+9.png) | ![](output\noise_style_loss\noise2_starry_night_starry_night_conv1+3+5+7+9.png) | ![](output\noise_style_loss\texture_image(noise1)_starry_night_starry_night_conv1+3+5+7+9.png) | ![](output\noise_style_loss\texture_image(noise2)_starry_night_starry_night_conv1+3+5+7+9.png) |
| conv6, conv7, conv8, conv9, conv10     | ![](output\noise_style_loss\noise1_starry_night_starry_night_conv6-10.png) | ![](output\noise_style_loss\noise2_starry_night_starry_night_conv6-10.png) | ![](output\noise_style_loss\texture_image(noise1)_starry_night_starry_night_conv6-10.png) | ![](output\noise_style_loss\texture_image(noise2)_starry_night_starry_night_conv6-10.png) |
| conv11, conv12, conv13, conv14, conv15 | ![](output\noise_style_loss\noise1_starry_night_starry_night_conv11-15.png) | ![](output\noise_style_loss\noise2_starry_night_starry_night_conv11-15.png) | ![](output\noise_style_loss\texture_image(noise1)_starry_night_starry_night_conv11-15.png) | ![](output\noise_style_loss\texture_image(noise2)_starry_night_starry_night_conv11-15.png) |



### Part 3  Style Transfer

#### 3.1 Tune the hyper-parameters

| ![](output\style_transfer\content_image_frida_kahlo_frida_kahlo_conv1+3+5+7+9.png) | ![](output\style_transfer\style_image_picasso_picasso_conv1+3+5+7+9.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\style_transfer\output_from_content_picasso_picasso_conv1+3+5+7+9_s  1000_c     1.png) | ![](output\style_transfer\output_from_content_picasso_picasso_conv1+3+5+7+9_s 10000_c     1.png) |
| ![](output\style_transfer\output_from_content_picasso_picasso_conv1+3+5+7+9_s100000_c     1.png) | ![](output\style_transfer\output_from_content_picasso_picasso_conv1+3+5+7+9_s1000000_c     1.png) |

#### 



| ![](output\style_transfer\content_image_frida_kahlo_frida_kahlo_conv1+3+5+7+9_s 10000_c1.png) | ![](output\style_transfer\style_image_frida_kahlo_frida_kahlo_conv1+3+5+7+9_s100000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\style_transfer\output_from_content_frida_kahlo_frida_kahlo_conv1+3+5+7+9_s  1000_c1.png) | ![](output\style_transfer\output_from_content_frida_kahlo_frida_kahlo_conv1+3+5+7+9_s 10000_c1.png) |
| ![](output\style_transfer\output_from_content_frida_kahlo_frida_kahlo_conv1+3+5+7+9_s100000_c1.png) | ![](output\style_transfer\output_from_content_frida_kahlo_frida_kahlo_conv1+3+5+7+9_s1000000_c1.png) |

| ![](output\style_transfer\content_image_starry_night_starry_night_conv1+3+5+7+9_s1000000_c1.png) | ![](output\style_transfer\style_image_starry_night_starry_night_conv1+3+5+7+9_s1000000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\style_transfer\output_from_content_starry_night_starry_night_conv1+3+5+7+9_s001000_c1.png) | ![](output\style_transfer\output_from_content_starry_night_starry_night_conv1+3+5+7+9_s010000_c1.png) |
| ![](output\style_transfer\output_from_content_starry_night_starry_night_conv1+3+5+7+9_s100000_c1.png) | ![](output\style_transfer\output_from_content_starry_night_starry_night_conv1+3+5+7+9_s1000000_c1.png) |



#### 3.2  Optimized results

From the result, the performance differs with different combinations of content image and style image. But mostly `100000` and `1000000` work the best.

| ![](output\style_transfer\output_from_content_picasso_picasso_conv1+3+5+7+9_s100000_c     1.png) | ![](output\style_transfer\output_from_content_frida_kahlo_frida_kahlo_conv1+3+5+7+9_s100000_c1.png) | ![](output\style_transfer\output_from_content_starry_night_starry_night_conv1+3+5+7+9_s1000000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



#### 3.3 Noise & Content image comparison 

By comparison we can see the output images from noise are more texture-liked. There are elements from original content image maintained, but still very vague.

| From noise                                                   | From content                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\style_transfer\output_from_noise_the_scream_the_scream_conv1+3+5+7+9_s1000000_c     1.png) | ![](output\style_transfer\output_from_content_the_scream_the_scream_conv1+3+5+7+9_s1000000_c     1.png) |
| ![](output\style_transfer\output_from_noise_starry_night_starry_night_conv1+3+5+7+9_s1000000_c1.png) | ![](output\style_transfer\output_from_content_starry_night_starry_night_conv1+3+5+7+9_s1000000_c1.png) |



#### 3.4 Additional results (with time compared)

Generally speaking, generating the style-transferred image from content image is approx. a quarter faster than from noise.



I took this photo in the Versailles Palace in Paris last December. Very impressive and I really recommend everyone to sightsee there :)

| ![](output\additional\content_image_klimt_gold_klimt_gold_feedforward, conv1+3+5+7+9_s100000_c1.png) | ![](output\additional\style_image_klimt_gold_klimt_gold_original, conv1+3+5+7+9_s1000000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\additional\output_from_noise_klimt_gold_klimt_gold_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\output_from_content_klimt_gold_klimt_gold_conv1+3+5+7+9_s1000000_c1.png) |

This photo is also taken by me! I was standing on the left bank of the Seine river, in front of the Musée d'Orsay in Paris. The style image, "Starry Night Over the Rhone", is one of Van Gogh's collections in Musée d'Orsay. When I saw the Seine river after visiting the museum, I immediately recalled this painting. 

| ![](output\additional\content_image_vangogh_river_vangogh_river_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\style_image_vangogh_river_vangogh_river_conv1+3+5+7+9_s1000000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\additional\output_from_noise_vangogh_river_vangogh_river_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\output_from_content_vangogh_river_vangogh_river_conv1+3+5+7+9_s1000000_c1.png) |

This style image is painted by Klimt. He is one of my favorite artists, I really recommend everyone to check his collections out.

| ![](output\additional\content_image_klimt_flower_klimt_flower_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\style_image_klimt_flower_klimt_flower_conv1+3+5+7+9_s1000000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\additional\output_from_noise_klimt_flower_klimt_flower_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\output_from_content_klimt_flower_klimt_flower_conv1+3+5+7+9_s1000000_c1.png) |



### Part 4  Bells & Whistles 

#### 4.1 Stylize image from the previous homework. 

I, again, became the sun.

| ![](output\additional\content_image_sunflower_sunflower_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\style_image_sunflower_sunflower_conv1+3+5+7+9_s1000000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\additional\output_from_noise_sunflower_sunflower_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\output_from_content_sunflower_sunflower_conv1+3+5+7+9_s1000000_c1.png) |

| ![](output\additional\content_image_kanagawa_kanagawa_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\style_image_kanagawa_kanagawa_conv1+3+5+7+9_s1000000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](output\additional\output_from_noise_kanagawa_kanagawa_conv1+3+5+7+9_s1000000_c1.png) | ![](output\additional\output_from_content_kanagawa_kanagawa_conv1+3+5+7+9_s1000000_c1.png) |

### 

#### 4.2 Implemented my own *cropping section*

###### Step 1

The basic idea is to **keep the content image the original size**. Therefore, the first step we are going to do is to check if the style image is large enough. If it is, we can directly jump to cropping. (If the style image is way too large, we can also indicate an upper limit of the ratio between the size of the style image and the content image, and then resize the style image to a reasonably smaller scale)



###### Step 2

Then, for those style images which are not large enough, we are going to check the ratio of the width of the style image with the width of the content image, as well as the ratio of lengths of them. 



###### Step 3

When one of the ratios `>=1`, we can consider that edge is long enough, and we will expand the style image with the ratio of another edge.

When both of the ratios `<1`, we will take the edge with the smaller ratio as the divider to expand the style image. When doing this, the other edge is automatically expanded to a valid length at the same time.

```python
width_ratio = style_width / content_width
height_ratio = style_height / content_height
style_ratio = style_height / style_width

# Step 1
if width_ratio >= 1 and height_ratio >= 1:
    pass

# Step 2
elif width_ratio < 1 and height_ratio >= 1:
    new_width = content_width
    new_height = int(new_width * style_ratio) + 1
    style_img = F.resize(style_img, (new_height, new_width))
    
# Step 3
elif width_ratio >= 1 and height_ratio < 1:
    new_height = content_height
    new_width = int(new_height / style_ratio) + 1
    style_img = F.resize(style_img, (new_height, new_width))
else:
    ratio = min(width_ratio, height_ratio)
    new_height = int(style_height / ratio) + 1
    new_width = int(style_width / ratio) + 1
    style_img = F.resize(style_img, (new_height, new_width))
```



###### Step 4

Finally, we can crop the image. We could start from the top left, or we can indicate any valid start point that is not out of range.

```python
top, left, height, width = 0, 0, content_height, content_width
```

 We could also use random crop to get more diverse results, for that synthesized image varies according to different style image input.

```python
from torchvision.transforms import RandomCrop
top, left, height, width = RandomCrop.get_params(style_img, (content_height, content_width))
```

Then, we manipulate  the style image:

```python
style_img = F.crop(style_img, top=top, left=left, height=height, width=width)
```

### 

#### 4.3 Tried to use a feedforward network to output style transfer results directly

| ![](output\additional\output_from_content_klimt_gold_klimt_gold_original, conv1+3+5+7+9_s100000_c1.png) | ![](output\additional\output_from_content_klimt_gold_klimt_gold_feedforward, conv1+3+5+7+9_s100000_c1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

