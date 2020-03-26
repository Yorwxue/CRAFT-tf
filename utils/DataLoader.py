import os
import re
import numpy as np


def SynLoader(dataset_path):
    pass


def CTWLoader(dataset_path):
    return 0


def TTLoader(dataset_path):
    data_dict = {"Train": list(), "Test": list()}

    for task in ["Train", "Test"]:
        data_path = os.path.abspath(os.path.join(dataset_path, "images", task))
        label_path = os.path.abspath(os.path.join(dataset_path, "labels", task))
        pseudo_label_path = os.path.abspath(os.path.join(dataset_path, "pseudo", task))
        data_path_list = os.listdir(data_path)
        for file_name in data_path_list:
            try:
                brief_filename = file_name.split(".")[0]
                img_path = os.path.join(data_path, file_name)

                label_filename = "poly_gt_%s.txt" % brief_filename
                pseudo_label_filename = "res_%s.txt" % brief_filename
                with open(os.path.join(label_path, label_filename), "r") as fr:
                    raws = fr.readlines()
                    word_boxes = list()
                    words = list()
                    char_boxes_list = list()
                    confidence_list = list()
                    for raw_line in raws:
                        raw_list = raw_line.split(", ")
                        word_boxes.append(getWordBox(raw_list))
                        words.append(raw_list[3].split(":")[1].replace("u'", "").replace("'", "").replace("\\", "").replace("[", "").replace("]", "").replace("\n", ""))
                        confidence_list.append(None)
                char_boxes_list.append(getCharBox(os.path.join(pseudo_label_path, pseudo_label_filename)))

                data_dict[task].append([img_path, word_boxes, words, char_boxes_list, confidence_list])
            except Exception as e:
                print(e)
                print("%s" % file_name)

    return data_dict["Train"], data_dict["Test"]


def minRect(word_box):
    xmin = np.min(word_box["x"])
    ymin = np.min(word_box["y"])
    xmax = np.max(word_box["x"])
    ymax = np.max(word_box["y"])
    return [[xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax]]


def getWordBox(box_string):
    word_box = dict()
    word_box["x"] = np.asarray((re.match("[xy]: \[\[ ?(.*)\]\]", box_string[0])).group(1).replace("   ", " ").replace("  ", " ").lstrip().split(" "), dtype=np.int)
    word_box["y"] = np.asarray((re.match("[xy]: \[\[ ?(.*)\]\]", box_string[1])).group(1).replace("   ", " ").replace("  ", " ").lstrip().split(" "), dtype=np.int)
    box = minRect(word_box)
    return box


def getCharBox(file_name):
    char_boxes = list()
    if os.path.exists(file_name):
        with open(file_name, "r") as fr:
            raw = fr.readlines()
            for raw_line in raw:
                char_box = raw_line.replace("\n", "").split(",")
                char_box = np.asarray(char_box)
                char_boxes.append(char_box)
        return char_boxes
    else:
        return list()


if __name__ == "__main__":
    train_real_data_list, test_data_list = TTLoader("../dataset/Total-Text/")

    # sample data index
    data_idx = 0

    img_path = train_real_data_list[data_idx][0]
    word_boxes = train_real_data_list[data_idx][1]
    words = train_real_data_list[data_idx][2]
    char_boxes_list = train_real_data_list[data_idx][3]
    confidence_list = train_real_data_list[data_idx][4]

    import cv2
    img = cv2.imread(img_path)

    from utils.box_util import reorder_points
    # word boxes
    # points = [reorder_points(np.asarray([[coord[0][0], coord[1][0]], [coord[0][1], coord[1][1]], [coord[0][2], coord[1][2]], [coord[0][3], coord[1][3]]])) for coord in word_boxes]
    # char boxes
    points = list()
    for point in char_boxes_list:
        pass

    points = [np.reshape(pts, (-1, 1, 2)) for pts in points]
    cv2.polylines(img, points, True, (0, 255, 255))
    cv2.imwrite("../result/sample_data_test.jpg", img)
