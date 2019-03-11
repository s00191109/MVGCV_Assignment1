import argparse
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter as gauss_filter

"""
@Author:            Enda McManemy  
@Student Nunber:    S00191109
@From:              Feb 2019  
@Code Derived from: https://github.com/schroeji/HarrisCorner/blob/master/main.py
@Description:       POC Code to experiment with four implementations of Harris Corners
                    Namely Harris-1988, Shi-Tomasi 1992, Triggs-2004 and  Brown
"""

# The unc path to the image file been processed
FilePath = 'C:\\Users\\emcmane\\Pictures\\cubes4.png'

#The Harris corner based algorithm to utilize
#Specify one of "harmonic", "harris", "shi-tomasi", "triggs"
ALGORITHM = 'harris'

# kernel window size
DELTA_X = 3
DELTA_Y = 3

# trace scaling factor
K = 0.06

# offset added to the trace of the M matrix when using the harmonic mean to prevent numerical instability
HARMONIC_OFFSET = 10e-6
# Threshold multiplier for R value
CORNER_THRESHOLD_MULTIPLIER = 0.2

def draw_dot(draw, x, y):
    """
    Draws a dot at the point x, y
    """
    r = 1
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0, 255))


def grey_scale(image):
    """
    Converts a colour image to a grey scale image
    """

    # Converting the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Conversion to float is a prerequisite for the algorithm
    gray_img = np.float32(gray_img)
    return gray_img


def get_derivatives(image):
    """
    Calculates and returns the products of the derivatives.
    i.e. Ixx, Ixy, Iyy
    """
    # first order derivatives
    Iy, Ix = np.gradient(image)
    # product of derivatives
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    return Ixx, Ixy, Iyy


def calc_tensor(Ixx, Ixy, Iyy, x, y):
    """
    Calculates and returns the structure tensor M.
    """
    # sum over window
    Sxx = sum(gauss_filter(Ixx[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1], 2).ravel())
    Sxy = sum(gauss_filter(Ixy[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1], 2).ravel())
    Syy = sum(gauss_filter(Iyy[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1], 2).ravel())

    M = np.asarray([[Sxx, Sxy], [Sxy, Syy]])
    return M


def generic_corner_detection(grey_scale_image, algorithm):
    """
    Performs corner detection using specified algorithm on the provided image.
    Returns a list of corners with inverted coordinates i.e. (x, y) corresponds
    to row x and column y in the image.
    """
    print("Started corner detection...")
    size_x = grey_scale_image.shape[0]
    size_y = grey_scale_image.shape[1]
    print("Calculating derivatives...")
    Ixx, Ixy, Iyy = get_derivatives(grey_scale_image)
    print("Calculating r_values for each pixel based on supplied algorithm...".format(algorithm))
    r_values = np.zeros(grey_scale_image.shape)
    # Iterate the Horizontal Pixels
    for x in range(DELTA_X, size_x - DELTA_X):
        # Iterate the Vertical Pixels
        for y in range(DELTA_Y, size_y - DELTA_Y):
            # Get the auto correlation Matrix
            M = calc_tensor(Ixx, Ixy, Iyy, x, y)
            # calculate the r_value i.e. the harris corner function
            if algorithm == "harris":
                r = np.linalg.det(M) - K*np.trace(M)**2.0
            elif algorithm == "harmonic":
                r = np.linalg.det(M)/(np.trace(M) + HARMONIC_OFFSET)
            elif algorithm == "shi-tomasi":
                lambs, _ = np.linalg.eig(M)
                if lambs[0] <= lambs[1]:
                    r = lambs[0]
                elif lambs[0] > lambs[1]:
                    r = lambs[1]
            elif algorithm == "triggs":
                lambs, _ = np.linalg.eig(M)
                r = lambs[0] - K*lambs[1]
            # store values for R at each pixel to an array
            r_values[x, y] = r
    print("Thresholding and nonmax supression...")
    max_r = max(r_values.ravel())
    list_of_corners = []
    # thresholding and nonmax supression
    for x in range(DELTA_X, size_x - DELTA_X):
        for y in range(DELTA_Y, size_y - DELTA_Y):
            max_in_window = max(r_values[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].ravel())
            # only use those r_values that are bigger than the threshold
            # and are the maximum in their respective window
            # if both are filfilled we found a corner
            if (r_values[x, y] >= max_r * CORNER_THRESHOLD_MULTIPLIER) and (r_values[x, y] == max_in_window):
                list_of_corners.append((x, y))
    # return all found corners
    return list_of_corners


def detect_interest_points(img):
    """
    Locates and draws the interest points in an image and displays the result.
    """

    corner_lists = []
    M = []

    print("Converting to grey scale...")
    grey_scale_image = grey_scale(img)
    corner_lists.append(generic_corner_detection(grey_scale_image, ALGORITHM))
    print("Found {} corners in image {}.".format(len(corner_lists[-1]), 1))
    print("Coloring corners...")
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    for (x, y) in corner_lists[-1]:
        draw_dot(draw, y, x)

    im.show()

def main():
    """
    Main function.
    """

    if FilePath == "":
        print("Please specify an Image file using the Constant FilePath.")
    elif ALGORITHM not in ["harmonic", "harris", "shi-tomasi", "triggs"]:
        print("Invalid algorithm specified for Constant ALGORITHM.")
    else:
        # load image into 3d array
        print(FilePath)
        image1 = np.asarray(Image.open(FilePath))
        print("Read image 1: {0}x{1} pixels.".format(*image1.shape))

        detect_interest_points(image1)

if __name__ == "__main__":
    main()

