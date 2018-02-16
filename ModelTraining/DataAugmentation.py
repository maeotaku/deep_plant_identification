
from __future__ import print_function

import os
import math

import numpy as np
#import theano
#import theano.tensor as T
#import lasagne

from scipy import misc

from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate

from random import randint

from enum import Enum

class AugmentationTypes(Enum):
    MirrorVertical=1,
    MirrorHorizontal=2,
    MirrorBoth=3,
    RandomClip=4,
    Rotation=5

class DLDataAug():

    '''
    def apply_augmentation(img, type):
        if type == AugmentationTypes.MirrorBoth:
            return DLDataAug.mirror_both(img)
        if type == AugmentationTypes.MirrorVertical:
            return DLDataAug.mirror_vertical(img)
        if type == AugmentationTypes.MirrorHorizontal:
            return DLDataAug.mirror_horizontal(img)

        #show_img(DLDataAug.mirror_vertical(images[2]))
        #show_img(DLDataAug.rotate(images[2], randint(30, 290)))
        #show_img(DLDataAug.clipped_zoom(images[2], 3))
        #crops = DLDataAug.crop(images[2], 3, 256, 0.2, 0.90)
        #show_img( np.hstack(crops) )
    '''

    @staticmethod
    def resize_img(img, img_size):
        try:
            resized = misc.imresize(img, [img_size, img_size], 'bilinear', mode=None)
            return resized
        except Exception as e:
            print("Cannot resize image", e)
            return None

    @staticmethod
    def mirror_vertical(batch):
        return np.fliplr(batch)

    @staticmethod
    def mirror_horizontal(batch):
        return np.flipup(batch)

    @staticmethod
    def mirror_both(batch):
        return mirror_horizontal(mirror_vertical(batch))

    @staticmethod
    def random_crop(img, perc_w, perc_h):
        wi = img.shape[1]
        hi = img.shape[2]
        wn = int(wi * perc_w)
        hn = int(hi * perc_h)
        xn = randint(0,hi-hn)
        yn = randint(0,wi-wn)
        return img[:, xn : xn + wn, xn : xn + wn]

    @staticmethod
    def crop(img, times, new_size, perc_min, perc_max):
        crops = np.zeros((times, 3, new_size, new_size), dtype='uint8')
        cont=0
        while cont < times:
            perc_w = float(randint( int(perc_min * 100), int(perc_max * 100)) / 100.0)
            perc_h = float(randint( int(perc_min * 100), int(perc_max * 100)) / 100.0)

            cropped = DLDataAug.random_crop(img, perc_w, perc_h)
            cropped = DLDataAug.resize_img(cropped, new_size)
            cropped = cropped.transpose()
            crops[cont, :, :, :] = cropped
            cont+=1
        return crops

    @staticmethod
    def rotate(img, angle):
        return rotate(img, angle, axes=(2, 1), reshape=False)

    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    @staticmethod
    def rotated_rect_with_max_area(w, h, angle):
        if w <= 0 or h <= 0:
            return 0,0

        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
        return wr,hr

    @staticmethod
    def rotate_and_crop(img, degrees):
        w = img.shape[1]
        h = img.shape[2]
        radians = degrees * (math.pi / 180.0)
        wr, hr = DLDataAug.rotated_rect_with_max_area(w, h, radians)
        x = (w - wr) / 2
        y = (h - hr) / 2
        rimg = DLDataAug.rotate(img, degrees)
        return rimg[:, x:x+wr, y:y+hr]
