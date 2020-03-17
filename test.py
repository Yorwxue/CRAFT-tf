import tensorflow as tf
import datetime
import argparse
import os
import cv2
import time
import numpy as np

from craft import CRAFT
import file_utils
import imgproc

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
args = parser.parse_args()


# def test_net(net_out_layers, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
#     t0 = time.time()
#
#     # resize
#     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
#     ratio_h = ratio_w = 1 / target_ratio
#
#     # preprocessing
#     x = imgproc.normalizeMeanVariance(img_resized)
#     x = np.expand_dims(x, axis=0)  # [h, w, c] to [b, h, w, c]
#
#     # forward pass
#     y, feature =
#
#     # make score and link map
#     score_text = y[0,:,:,0].cpu().data.numpy()
#     score_link = y[0,:,:,1].cpu().data.numpy()
#
#     # refine link
#     if refine_net is not None:
#         with torch.no_grad():
#             y_refiner = refine_net(y, feature)
#         score_link = y_refiner[0,:,:,0].cpu().data.numpy()
#
#     t0 = time.time() - t0
#     t1 = time.time()
#
#     # Post-processing
#     boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
#
#     # coordinate adjustment
#     boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
#     polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
#     for k in range(len(polys)):
#         if polys[k] is None: polys[k] = boxes[k]
#
#     t1 = time.time() - t1
#
#     # render results (optional)
#     render_img = score_text.copy()
#     render_img = np.hstack((render_img, score_link))
#     ret_score_text = imgproc.cvt2HeatmapImg(render_img)
#
#     if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
#
#     return boxes, polys, ret_score_text


if __name__ == "__main__":
    net = CRAFT()

    tf.keras.models.Model()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    pass

    image = np.zeros((224, 224, 3))
    images = np.expand_dims(image, axis=0)
    with tf.GradientTape() as tape:
        logits = net(images)
    pass

    # image_list, _, _ = file_utils.get_files(args.test_folder)
    # result_folder = './result/'
    # if not os.path.isdir(result_folder):
    #     os.mkdir(result_folder)
    #
    # # load data
    # for k, image_path in enumerate(image_list):
    #     print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
    #     image = imgproc.loadImage(image_path)

    #     bboxes, polys, score_text = test_net(net_out_layers, image, args.text_threshold, args.link_threshold, args.low_text,
    #                                          args.cuda, args.poly, refine_net)
    #
    #     # save score text
    #     filename, file_ext = os.path.splitext(os.path.basename(image_path))
    #     mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    #     cv2.imwrite(mask_file, score_text)
    #
    #     file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
    #
    # print("elapsed time : {}s".format(time.time() - t))
