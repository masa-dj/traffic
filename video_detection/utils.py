import numpy as np


# Checking the percentage of object inside of ROI
def rect_in_zone(x1, y1, x2, y2, mask):
    h, w = mask.shape
    # clipping
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, w), min(y2, h)

    if x2c <= x1c or y2c <= y1c:
        return 0.0

    box_area = (x2 - x1) * (y2 - y1)
    if box_area == 0:
        return 0.0

    return np.count_nonzero(mask[y1c:y2c, x1c:x2c]) / box_area
