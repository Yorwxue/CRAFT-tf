import tensorflow as tf
import datetime
import argparse
import os
import cv2
import numpy as np
import datetime

from craft import CRAFT
from utils.img_util import resize_aspect_ratio, img_normalize, score_to_heat_map, load_image
from utils.file_utils import get_files, saveResult
from utils.craft_utils import getDetBoxes,  adjustResultCoordinates

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--test_folder', default='./images/', type=str, help='folder path to input images')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--canvas_size', default=768, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default="PATH/TO/YOUR/REFINER/MODEL", type=str, help='pretrained refiner model')
parser.add_argument('--weight_dir', type=str, default=r"./weights/", help="directory of model weights")

args = parser.parse_args()


def test_net(net, image, text_threshold, link_threshold, low_text, poly, refine_net):
    # t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = img_normalize(img_resized)
    x = np.expand_dims(x, axis=0)  # [h, w, c] to [b, h, w, c]

    # forward pass
    y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0]
    score_link = y[0, :, :, 1]

    # refine link
    if refine_net is not None:
        # TODO
        pass

    # t0 = time.time() - t0
    # t1 = time.time()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.numpy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = score_to_heat_map(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


if __name__ == "__main__":
    net = CRAFT(input_shape=(args.canvas_size, args.canvas_size, 3))
    checkpoint = tf.train.Checkpoint(model=net)
    checkpoint_dir = tf.train.latest_checkpoint(args.weight_dir)
    # checkpoint_dir = os.path.join(args.weight_dir, "ckpt-10")
    checkpoint.restore(checkpoint_dir)
    print("Restored from %s" % checkpoint_dir)
    # prefix_filename = datetime.datetime.now().strftime('%m%d_%H:%M:%S')
    prefix_filename = checkpoint_dir.split("/")[-1]

    # if not os.path.exists("./logs/fit/"):
    #     os.makedirs("logs/fit/")
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    image_list, _, _ = get_files(args.test_folder)
    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    # LinkRefiner
    refine_net = None
    # if args.refine:
    #     # TODO
    #     pass

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = load_image(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/%s_res_" % prefix_filename + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder, prefix=prefix_filename)

    # print("elapsed time : {}s".format(time.time() - t))
