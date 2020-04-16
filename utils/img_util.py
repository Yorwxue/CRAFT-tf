"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import cv2
import numpy as np
from skimage import io
from utils.box_util import cal_affinity_boxes
from utils.train_utils import random_crop

# RGB
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
NORMALIZE_VARIANCE = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def load_image(img_path):
    """
    Load an image from file.
    :param img_path: Image file path, e.g. ``test.jpg`` or URL.
    :return: An RGB-image MxNx3.
    """
    img = io.imread(img_path)
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.array(img)

    return img


def img_normalize(src):
    """
    Normalize a RGB image.
    :param src: Image to normalize. Must be RGB order.
    :return: Normalized Image
    """
    img = src.copy().astype(np.float32)

    img -= NORMALIZE_MEAN
    img /= NORMALIZE_VARIANCE
    return img


def img_unnormalize(src):
    """
    Unnormalize a RGB image.
    :param src: Image to unnormalize. Must be RGB order.
    :return: Unnormalized Image.
    """
    img = src.copy()

    img *= NORMALIZE_VARIANCE
    img += NORMALIZE_MEAN

    return img.astype(np.uint8)


# def img_resize(src, ratio, max_size, interpolation):
#     """
#     Resize image with a ratio.
#     :param src: Image to resize.
#     :param ratio: Scaling ratio.
#     :param max_size: Maximum size of Image.
#     :param interpolation: Interpolation method. See OpenCV document.
#     :return: dst: Resized image.
#              target_ratio: Actual scaling ratio.
#     """
#     img = src.copy()
#     height, width, channel = img.shape
#
#     target_ratio = min(max_size / max(height, width), ratio)
#     target_h, target_w = int(height * target_ratio), int(width * target_ratio)
#     dst = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
#
#     return dst, target_ratio


def resize_aspect_ratio(img, square_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    # if target_size > square_size:
    target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    # target_h32, target_w32 = target_h, target_w
    # if target_h % 32 != 0:
    #     target_h32 = target_h + (32 - target_h % 32)
    # if target_w % 32 != 0:
    #     target_w32 = target_w + (32 - target_w % 32)
    # resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    # resized[0:target_h, 0:target_w, :] = proc
    # target_h, target_w = target_h32, target_w32

    # padding
    resized = np.zeros((square_size, square_size, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = square_size, square_size

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def score_to_heat_map(score):
    """
    Convert region score or affinity score to heat map.
    :param score: Region score or affinity score.
    :return: Heat map.
    """
    heat_map = (np.clip(score, 0, 1) * 255).astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    return heat_map


def create_affinity_box(boxes):
    affinity_boxes = cal_affinity_boxes(boxes)
    return affinity_boxes


def create_score_box(boxes_list):
    region_box_list = list()
    affinity_box_list = list()

    for boxes in boxes_list:
        region_box_list.extend(boxes)
        if len(boxes) > 0:
            affinity_box_list.extend(create_affinity_box(boxes))

    return region_box_list, affinity_box_list


def load_sample(img_path, img_size, raw_word_boxes, raw_char_boxes_list, crop_ratio=(0.1, 0.3)):
    raw_img = load_image(img_path)
    min_crop_ratio = crop_ratio[0]
    max_crop_ratio = crop_ratio[1]

    img, word_boxes, char_boxes_list = random_crop(raw_img, raw_word_boxes, raw_char_boxes_list, (min_crop_ratio, max_crop_ratio))

    # redo sample data while this data is broken
    while len(char_boxes_list) == 0 or len(word_boxes) == 0:
        print("img: " + img_path + " can't find boxes")
        min_crop_ratio = np.min([min_crop_ratio+0.1, 1])
        max_crop_ratio = np.min([max_crop_ratio+0.1, 1])
        img, word_boxes, char_boxes_list = random_crop(raw_img, word_boxes, char_boxes_list, (min_crop_ratio, max_crop_ratio))


    height, width = img.shape[:2]
    # ratio = img_size / max(height, width)
    # target_height = int(height * ratio)
    # target_width = int(width * ratio)
    # img = cv2.resize(img, (target_width, target_height))
    #
    # normalized_img = img_normalize(img)
    # # padding
    # img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    # img[:target_height, :target_width] = normalized_img
    img, ratio, size_heatmap = resize_aspect_ratio(img, img_size)
    target_height = int(height * ratio)
    target_width = int(width * ratio)
    img = img_normalize(img)

    word_boxes = [[[int(x * ratio), int(y * ratio)] for x, y in box] for box in word_boxes]

    if len(char_boxes_list) == 0:
        return img, word_boxes, char_boxes_list, [], [], (target_width, target_height)

    char_boxes_list = [[[[int(x * ratio), int(y * ratio)] for x, y in char_box] for char_box in word_box] for word_box in char_boxes_list]
    region_box_list, affinity_box_list = create_score_box(char_boxes_list)

    return img, word_boxes, char_boxes_list, region_box_list, affinity_box_list, (target_width, target_height)


def to_heat_map(src):
    dst = (np.clip(src, 0, 1) * 255).astype(np.uint8)
    dst = cv2.applyColorMap(dst, cv2.COLORMAP_JET)
    return dst


def drawPoly(src, points_list):
    """

    Args:
        src: input image
        points_list: several points as a group, and this is a list of groups

    Returns: an image

    """
    image = src.copy()
    reshaped_points_list = list()
    for points in points_list:
        points = np.asarray(points, dtype=np.int)
        points = np.reshape(points, (-1, 2))
        reshaped_points_list.append(points)

    cv2.polylines(image, reshaped_points_list, True, (0, 255, 255))
    return image
