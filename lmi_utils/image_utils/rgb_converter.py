"""
integer converter: converts between integers and rgb encodings [red,green,blue]
image converter: converts images
"""

import numpy as np

TWO_TO_TWENTYFORUTH_MINUS_ONE = 16777215


def convert_to_rgb(an_int):
    blue = an_int & 255
    green = (an_int >> 8) & 255
    red = (an_int >> 16) & 255
    return np.uint8(red), np.uint8(green), np.uint8(blue)

def convert_array_to_rgb(an_array_of_ints):
    blue = an_array_of_ints & 255
    green = (an_array_of_ints >> 8) & 255
    red = (an_array_of_ints >> 16) & 255
    rgb_image=np.dstack((red,green,blue)).astype(np.uint8)
    return rgb_image

def convert_to_rainbow(an_int):
    COLOR_BIN=TWO_TO_TWENTYFORUTH_MINUS_ONE//7
    slope=255/COLOR_BIN
    try: 
        # from black...ramp-up blue in first bin
        if an_int<COLOR_BIN:
            x=an_int
            blue=x*slope
            green=0
            red=0
        # ramp-up green in second bin
        elif an_int>=COLOR_BIN and an_int<COLOR_BIN*2:
            x=an_int-COLOR_BIN
            blue=255
            green=x*slope
            red=0
        # ramp-down blue in third bin
        elif an_int>=COLOR_BIN*2 and an_int<COLOR_BIN*3:
            x=an_int-2*COLOR_BIN
            blue=255-x*slope
            green=255
            red=0
        # ramp-up red in the fourth bin
        elif an_int>=COLOR_BIN*3 and an_int<COLOR_BIN*4:
            x=an_int-3*COLOR_BIN
            blue=0
            green=255
            red=x*slope
        #ramp-down green in 5th bin
        elif an_int>=COLOR_BIN*4 and an_int<COLOR_BIN*5:
            x=an_int-4*COLOR_BIN
            blue=0
            green=255-x*slope
            red=255
        #ramp-up blue in 6th bin
        elif an_int>=COLOR_BIN*5 and an_int<COLOR_BIN*6:
            x=an_int-5*COLOR_BIN
            blue=x*slope
            green=0
            red=255
        #ramp-up green (white=highest)
        elif an_int>=COLOR_BIN*6 and an_int<COLOR_BIN*7:
            x=an_int-6*COLOR_BIN
            blue=255
            green=x*slope
            red=255
    except:
        print('Invald range.')

    return np.uint8(red), np.uint8(green), np.uint8(blue)

def convert_array_to_rainbow(an_array_of_ints):
    COLOR_BIN=TWO_TO_TWENTYFORUTH_MINUS_ONE//7
    slope=255/COLOR_BIN
    blue=np.zeros(an_array_of_ints.shape)
    green=np.zeros(an_array_of_ints.shape)
    red=np.zeros(an_array_of_ints.shape)
    try:
        # from black...ramp-up blue in first bin
        bin1_index=an_array_of_ints<COLOR_BIN
        blue[bin1_index]=an_array_of_ints[bin1_index]*slope
        # ramp-up green in second bin
        bin2_index=np.logical_and(an_array_of_ints>=COLOR_BIN,an_array_of_ints<COLOR_BIN*2)
        blue[bin2_index]=255
        green[bin2_index]=(an_array_of_ints[bin2_index]-COLOR_BIN)*slope
        # ramp-down blue in third bin
        bin3_index=np.logical_and(an_array_of_ints>=COLOR_BIN*2,an_array_of_ints<COLOR_BIN*3)
        blue[bin3_index]=255-(an_array_of_ints[bin3_index]-2*COLOR_BIN)*slope
        green[bin3_index]=255
        red[bin3_index]=0
        # ramp-up red in the fourth bin
        bin4_index=np.logical_and(an_array_of_ints>=COLOR_BIN*3,an_array_of_ints<COLOR_BIN*4)
        blue[bin4_index]=0
        green[bin4_index]=255
        red[bin4_index]=(an_array_of_ints[bin4_index]-3*COLOR_BIN)*slope
        #ramp-down green in 5th bin
        bin5_index=np.logical_and( an_array_of_ints>=COLOR_BIN*4,an_array_of_ints<COLOR_BIN*5)
        blue[bin5_index]=0
        green[bin5_index]=255-(an_array_of_ints[bin5_index]-4*COLOR_BIN)*slope
        red[bin5_index]=255
        #ramp-up blue in 6th bin
        bin6_index=np.logical_and(an_array_of_ints>=COLOR_BIN*5,an_array_of_ints<COLOR_BIN*6)
        blue[bin6_index]=(an_array_of_ints[bin6_index]-5*COLOR_BIN)*slope
        green[bin6_index]=0
        red[bin6_index]=255
        #ramp-up green (white=highest)
        bin7_index=np.logical_and(an_array_of_ints>=COLOR_BIN*6,an_array_of_ints<COLOR_BIN*7)
        blue[bin7_index]=255
        green[bin7_index]=(an_array_of_ints[bin7_index]-6*COLOR_BIN)*slope
        red[bin7_index]=255
    except:
        raise Exception('Invald range.')

    rgb_image=np.dstack((red,green,blue)).astype(np.uint8)
    return rgb_image


def convert_from_rgb(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    an_int = (red << 16) + (green << 8) + blue
    return an_int


def convert_greyscale_image_to_color(greyscale_img):
    """
    :encoding spans image min/max greyscale values [min-----range -----max], so quantization error = range/255
    :param greyscale_img: numpy array with ints greyscale values from 0 to 255
    :return: color_img: numpy array with 3 channel rbg encoded form of the greyscale input
    """
    if len(np.shape(greyscale_img)) > 2:
        return greyscale_img
        # return np.transpose(greyscale_img, (2,0,1))
    greyscale_y = np.arange(0, TWO_TO_TWENTYFORUTH_MINUS_ONE)
    greyscale_x = np.linspace(
        greyscale_img.min(), greyscale_img.max(), num=TWO_TO_TWENTYFORUTH_MINUS_ONE
    )
    greyscale_int = np.interp(greyscale_img, greyscale_x, greyscale_y).astype(
        np.int
    )
    color_img = np.zeros([3] + list(np.shape(greyscale_img)), dtype=np.uint8)
    num_rows, num_cols = np.shape(greyscale_img)
    for i in range(num_rows):
        for j in range(num_cols):
            color_img[:, i, j] = convert_to_rgb(greyscale_int[i, j])
    return color_img


def convert_greyscale_to_color_simple(greyscale_img):
    """
    :param greyscale_img: np array with ints greyscale values from 0 to 255
    :return: color_img: np array with 3 channel rbg which reproduces the greyscale image 3 times
    """
    if len(np.shape(greyscale_img)) > 2:
        return greyscale_img
        # return np.transpose(greyscale_img, (2,0,1))
    return np.concatenate([np.expand_dims(greyscale_img, 0)] * 3)
    

if __name__ == "__main__":
    import cv2
    ramp=np.arange(0,TWO_TO_TWENTYFORUTH_MINUS_ONE,50000)
    n=ramp.shape[0]
    x,y=np.meshgrid(ramp,ramp)
    x_rb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    y_rb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    x_rgb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    y_rgb=np.zeros([len(ramp),len(ramp),3],dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            print('[INFO] i:',str(i),', j:',str(j))
            x_rb[i,j]=convert_to_rainbow(x[i,j])
            y_rb[i,j]=convert_to_rainbow(y[i,j])
            x_rgb[i,j]=convert_to_rgb(x[i,j])
            y_rgb[i,j]=convert_to_rgb(y[i,j])
    
    cv2.imshow('x rainbow',cv2.cvtColor(x_rb,cv2.COLOR_RGB2BGR))
    cv2.imshow('y rainbow',cv2.cvtColor(y_rb,cv2.COLOR_RGB2BGR))
    cv2.imshow('x rgb',cv2.cvtColor(x_rgb,cv2.COLOR_RGB2BGR))
    cv2.imshow('y rgb',cv2.cvtColor(y_rgb,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

