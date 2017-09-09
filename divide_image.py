#!/usr/bin/env python

def divide_image(image, height_new, width_new):
    height, width, channels = image.shape
    if height < height_new or width < width_new:
        print("Error: in divide_image original dimensions are less than new ones")
        return [image.copy()]
    image_parts = []
    step_x = width_new / 2
    step_y = height_new / 2
    x_beg = 0; x_end = x_beg + width_new
    while x_end < width:
        y_beg = 0; y_end = y_beg + height_new
        while y_end < height:
            image_parts.append(image[y_beg:y_end, x_beg:x_end, :].copy())
            y_beg = y_end; y_end = y_beg + height_new

        y_end = height; y_beg = y_end - height_new
        image_parts.append(image[y_beg:y_end, x_beg:x_end, :].copy())

        x_beg = x_end; x_end = x_beg + width_new

    x_end = width; x_beg = x_end - width_new
    y_beg = 0; y_end = y_beg + height_new
    while y_end < height:
        image_parts.append(image[y_beg:y_end, x_beg:x_end, :].copy())
        y_beg = y_end; y_end = y_beg + height_new

    y_end = height; y_beg = y_end - height_new
    image_parts.append(image[y_beg:y_end, x_beg:x_end, :].copy())

    return image_parts
