import os
import re
import numpy as np
import json

# from utils.img_util import load_image, drawPoly


def SynLoader(dataset_path):
    pass


class TTLoader(object):
    def __init__(self, dataset_path):
        self.data_dict = {"Train": list(), "Test": list()}

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
                        confidence_list = list()
                        for raw_line in raws:
                            raw_list = raw_line.split(", ")
                            word_boxes.append(self.getWordBox(raw_list))
                            words.append(raw_list[3].split(":")[1].replace("u'", "").replace("'", "").replace("\\", "").replace("[", "").replace("]", "").replace("\n", ""))
                            confidence_list.append(None)
                    char_boxes_list = self.getPseudoBox(os.path.join(pseudo_label_path, pseudo_label_filename))

                    self.data_dict[task].append([img_path, word_boxes, words, char_boxes_list, confidence_list])
                except Exception as e:
                    print(e)
                    print("%s" % file_name)

    def get_dataset(self):
        return self.data_dict["Train"], self.data_dict["Test"]

    def getWordBox(self, box_string):
        word_box = dict()
        word_box["x"] = np.asarray((re.match("[xy]: \[\[ ?(.*)\]\]", box_string[0])).group(1).replace("   ", " ").replace("  ", " ").lstrip().split(" "), dtype=np.int)
        word_box["y"] = np.asarray((re.match("[xy]: \[\[ ?(.*)\]\]", box_string[1])).group(1).replace("   ", " ").replace("  ", " ").lstrip().split(" "), dtype=np.int)
        box = self.minRect(word_box)
        return box

    def getPseudoBox(self, file_name):
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

    def minRect(self, word_box):
        xmin = np.min(word_box["x"])
        ymin = np.min(word_box["y"])
        xmax = np.max(word_box["x"])
        ymax = np.max(word_box["y"])
        return [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]


class CTWLoader(object):
    def __init__(self, dataset_path):
        self.root_path = dataset_path
        self.test_set = list()
        self.train_set = list()

        with open(os.path.join(self.root_path, "test_cls.jsonl"), "r") as fr:
            raw = fr.readlines()
            for raw_line in raw:
                dict_line = json.loads(raw_line)
                img_path = os.path.abspath(os.path.join(self.root_path, "test", dict_line["file_name"]))
                char_boxes_list = [proposal["polygon"] for proposal in dict_line["proposals"]]
                word_boxes = list()
                words = list()
                confidence_list = list()
                self.test_set.append([img_path, word_boxes, words, char_boxes_list, confidence_list])

                # # show boxes
                # file_name = re.search(r'"file_name": "(.+)", "height"', raw_line).group(1)
                # file_path = os.path.join(self.root_path, "test", file_name)
                # proposals = json.loads(re.search(r'"proposals": (.+), "width"', raw_line).group(1))
                # image = load_image(file_path)
                # display = drawPoly(image, [proposal["polygon"] for proposal in proposals])
                # import cv2
                # cv2.imwrite("../result/sample_data_test.jpg", display)
                # print("Image saved at %s" % os.path.abspath("../result/sample_data_test.jpg"))

        with open(os.path.join(self.root_path, "train.jsonl"), "r") as fr:
            raw = fr.readlines()
            for raw_line in raw:
                dict_line = json.loads(raw_line)
                img_path = os.path.abspath(os.path.join(self.root_path, "train", dict_line["file_name"]))
                word_boxes = list()
                words = list()
                char_boxes_list = list()
                confidence_list = list()
                for annotation in dict_line["annotations"]:
                    word_box, word, char_boxes = self.charMerge(annotation)
                    word_boxes.append(word_box)
                    words.append(word)
                    char_boxes_list.append(char_boxes)
                    confidence_list.append(1)

                self.train_set.append([img_path, word_boxes, words, char_boxes_list, confidence_list])

                # # show box
                # file_path = os.path.join(self.root_path, "train", dict_line["file_name"])
                # image = load_image(file_path)
                # points_list = list()
                # for annotation in dict_line["annotations"]:
                #     for char_box in annotation:
                #         points_list.append(char_box["polygon"])
                # display = drawPoly(image, points_list)
                # import cv2
                # cv2.imwrite("../result/sample_data_test.jpg", display)
                # print("Image saved at %s" % os.path.abspath("../result/sample_data_test.jpg"))

    def charMerge(self, annotation):
        """

        Args:
            annotation: character-level word bbox, such as :
             ---    ---    ---    ---    ---
            | H |  | E |  | L |  | L |  | O |
             ---    ---    ---    ---    ---
            totally 5 character boxes
        Returns: (1) a list of coordinates of box order by: top-left, top-right, bottom-right, bottom-left
                 (2) word
                 (3) list of character boxes

        """
        char_box_list = list()
        points = list()
        word = ""
        for idx in range(len(annotation)):
            char_box = annotation[idx]["polygon"]
            char_box_list.append(char_box)
            points.extend(char_box)
            word += annotation[idx]["text"]
        points = np.asarray(points)
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1])
        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], word, char_box_list

    def get_dataset(self):
        return self.train_set, self.test_set


if __name__ == "__main__":
    # Total Text Dataset
    """
    TTdataset = TTLoader("../dataset/Total-Text/")
    train_real_data_list, test_data_list = TTdataset.get_dataset()

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
    # points_list = [reorder_points(np.asarray([[coord[0][0], coord[1][0]], [coord[0][1], coord[1][1]], [coord[0][2], coord[1][2]], [coord[0][3], coord[1][3]]])) for coord in word_boxes]
    # char boxes
    points_list = list()
    for char_box in char_boxes_list:
        points = np.asarray(char_box, dtype=np.int)
        points = np.reshape(points, (-1, 2))
        points_list.append(points)

    points_list = [np.reshape(pts, (-1, 1, 2)) for pts in points_list]
    cv2.polylines(img, points_list, True, (0, 255, 255))
    cv2.imwrite("../result/sample_data_test.jpg", img)
    """

    # A Large Chinese Text Dataset in the Wild
    CTWdataset = CTWLoader("../dataset/ctw/")
    print()
