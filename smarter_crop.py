def smart_crop(x, y, w, h, frame_height, frame_width, increment=10):
    height_increment = frame_height / increment
    width_increment = frame_width / increment

    endX = x + w
    endY = y + h

    newX = (x // width_increment) * width_increment
    newY = (y // height_increment) * height_increment

    if (endX // width_increment) * width_increment < endX:
        newendX = ((endX // width_increment) + 1) * width_increment
    else:
        newendX = (endX // width_increment) * width_increment

    if (endY // height_increment) * height_increment < endY:
        newendY = ((endY // height_increment) + 1) * height_increment
    else:
        newendY = (endY // height_increment) * height_increment

    newW = newendX - newX
    newH = newendY - newY

    return (int(newX), int(newY), int(newW), int(newH))