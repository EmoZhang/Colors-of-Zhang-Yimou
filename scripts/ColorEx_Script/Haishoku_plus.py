import os
from PIL import Image
from haishoku.haishoku import Haishoku


def new_image(mode, size, color):
    """ generate a new color block
        to generate the palette
    """
    new = Image.new(mode, size, color)
    return new


def joint_image(images, path, idx, t):
    """ generate the palette
        size: 50 x 400
        color_block_size: 50 x 50
    """
    palette = Image.new('RGB', (400, 20))

    # init the box position
    init_ul = 0

    for image in images:
        palette.paste(image, (init_ul, 0))
        init_ul += image.width

    palette.save(path + 'results/colors/' + t + '-%05d' % idx + '.png')


def saveDominant(image_path, path, idx):
    # get the dominant color
    dominant = Haishoku.getDominant(image_path)

    # generate colors boxes
    images = []
    dominant_box = new_image('RGB', (50, 20), dominant)
    for i in range(8):
        images.append(dominant_box)

    # save dominant color
    joint_image(images, path, idx, 'Dominant')


def savePalette(image_path, path, idx):
    # get the palette first
    palette = Haishoku.getPalette(image_path)

    # getnerate colors boxes
    images = []
    for color_mean in palette:
        w = color_mean[0] * 400
        color_box = new_image('RGB', (int(w), 20), color_mean[1])
        images.append(color_box)

    # generate and save the palette
    joint_image(images, path, idx, 'Palette')
