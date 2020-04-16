import numpy as np


def random_crop(img, word_boxes, char_boxes_list, crop_ratio=(0.1, 0.3)):
    """

    Args:
        img:
        word_boxes:  bounding boxes in square-shape
                     coordinates of each word box is ordered by top-left, top-right, bottom-right, bottom-left
        char_boxes_list:

    Returns: (1) cropped image
             (2) word-level boxes of cropped image
             (3) character-level boxes of cropped image

    """
    # decide the ratio of size of cropped image over size of raw image
    if crop_ratio[0] == crop_ratio[1]:
        random_ratio = crop_ratio[0]
    else:
        random_ratio = np.random.uniform(crop_ratio[0], crop_ratio[1])

    # random choose a box as center of cropped image
    random_chose = np.random.randint(0, len(word_boxes))

    # bias base on box size
    random_center_x = np.random.randint(
        np.min([word_boxes[random_chose][0][0], word_boxes[random_chose][3][0]]),
        np.max([word_boxes[random_chose][1][0], word_boxes[random_chose][2][0]]))
    random_center_y = np.random.randint(
        np.min([word_boxes[random_chose][0][1], word_boxes[random_chose][1][1]]),
        np.max([word_boxes[random_chose][2][1], word_boxes[random_chose][3][1]]))

    high, width = np.shape(img)[0:2]
    target_high = int(random_ratio * high)
    target_width = int(random_ratio * width)

    # bias for move box far from center
    random_center_x = random_center_x + np.random.randint(-target_width//4, target_width//4)
    random_center_y = random_center_y + np.random.randint(-target_high//4, target_high//4)

    # move center to avoid out of boundary
    if (random_center_x - target_width//2) < 0:
        random_center_x = random_center_x + target_width//2
    elif (random_center_x + target_width//2) >= width:
        random_center_x = random_center_x - target_width//2
    if (random_center_y - target_high//2) < 0:
        random_center_y = random_center_x + target_high//2
    elif (random_center_y + target_high//2) >= width:
        random_center_y = random_center_x - target_high//2

    # compute boundary
    img_boundary_xmin = np.max([0, random_center_x - int(target_width/2)])
    img_boundary_xmax = np.min([random_center_x + int(target_width/2), width])
    img_boundary_ymin = np.max([0, random_center_y - int(target_high/2)])
    img_boundary_ymax = np.min([random_center_y + int(target_high/2), high])
    target_img = img[
        img_boundary_ymin:img_boundary_ymax,
        img_boundary_xmin:img_boundary_xmax,
        :
    ]
    target_word_boxes = boxes_checker(word_boxes, (target_high, target_width), (random_center_x, random_center_y))
    target_char_boxes_list = list()
    for char_boxes in char_boxes_list:
        target_char_boxes = boxes_checker(char_boxes, (target_high, target_width), (random_center_x, random_center_y))
        if len(target_char_boxes) > 0:
            target_char_boxes_list.append(target_char_boxes)
    return target_img, target_word_boxes, target_char_boxes_list


def boxes_checker(boxes, new_img_shape, old_img_center):
    high, width = new_img_shape
    bias_x, bias_y = old_img_center
    target_boxes = list()
    boxes = np.asarray(boxes)
    for box in boxes:
        box[:, 0] = box[:, 0] - int(bias_x - width/2)
        box[:, 1] = box[:, 1] - int(bias_y - high/2)
        if np.min(box[:, 0]) > width: continue
        if np.min(box[:, 1]) > high: continue
        if np.max(box[:, 0]) < 0: continue
        if np.max(box[:, 1]) < 0: continue

        box[box[:, 0] >= width, 0] = width - 1
        box[box[:, 0] < 0, 0] = 0
        box[box[:, 1] >= high, 1] = high - 1
        box[box[:, 1] < 0, 1] = 0
        target_boxes.append(box)
    return target_boxes
