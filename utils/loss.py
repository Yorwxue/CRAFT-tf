import tensorflow as tf
import os
import numpy as np
import cv2


def ohem(loss, fg_mask, bg_mask, negative_ratio=3.):
    fg_num = tf.reduce_sum(fg_mask)
    bg_num = tf.reduce_sum(bg_mask)

    # neg_num = tf.maximum(tf.cast(fg_num * negative_ratio, dtype=tf.int32), tf.constant(10000, dtype=tf.int32))
    neg_num = tf.cast(fg_num * negative_ratio, dtype=tf.int32)
    neg_num = tf.minimum(tf.cast(bg_num, dtype=tf.int32), neg_num)
    neg_num = tf.minimum(tf.cast(1, dtype=tf.int32), neg_num)  # prevent that there are no fg_mask

    neg_loss = loss * bg_mask
    vals, _ = tf.nn.top_k(tf.reshape(neg_loss, shape=[-1]), k=neg_num)

    # # soft-mask
    # bg_bool_mask = tf.greater_equal(neg_loss, vals[-1])
    # soft_bg_mask = neg_loss * tf.cast(bg_bool_mask, dtype=tf.float32)
    # return soft_bg_mask

    # hard-mask
    bg_bool_mask = tf.cast(bg_mask, dtype=tf.bool)
    hard_bg_bool_mask = tf.logical_and(bg_bool_mask, tf.greater_equal(neg_loss, vals[-1]))
    hard_bg_mask = tf.cast(hard_bg_bool_mask, dtype=tf.float32)
    return hard_bg_mask


def batch_ohem(loss, fg_mask, bg_mask, negative_ratio=3.):
    return tf.map_fn(lambda x: ohem(x[0], x[1], x[2], negative_ratio), elems=[loss, fg_mask, bg_mask], dtype=tf.float32)


def save_log(preds, losses, gts, fg_masks, bg_masks, type, prefix):
    idx = 0
    for pred, loss, gt, fg_mask, bg_mask in zip(preds, losses, gts, fg_masks, bg_masks):
        pred = np.asarray(pred)
        loss = np.asarray(loss)
        gt = np.asarray(gt)
        fg_mask = np.asarray(fg_mask)
        bg_mask = np.asarray(bg_mask)
        pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        loss_norm = loss  # (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
        train_mask = fg_mask + bg_mask
        img_pred = np.transpose([pred_norm, pred_norm, pred_norm], (1, 2, 0)) * 255
        img_loss = np.transpose([loss_norm, loss_norm, loss_norm], (1, 2, 0)) * 255
        img_gt = np.transpose([gt, gt, gt], (1, 2, 0)) * 255
        img_mask = np.transpose([train_mask, train_mask, train_mask], (1, 2, 0)) * 255
        if not os.path.exists("logs"):
            os.makedirs("logs")
        cv2.imwrite("./logs/%s_%d_%s_img_pred.jpg" % (prefix, idx, type), img_pred)
        cv2.imwrite("./logs/%s_%d_%s_loss.jpg" % (prefix, idx, type), img_loss)
        cv2.imwrite("./logs/%s_%d_%s_gt.jpg" % (prefix, idx, type), img_gt)
        cv2.imwrite("./logs/%s_%d_%s_masked.jpg" % (prefix, idx, type), img_mask)
        idx += 1
    print("%s log saved." % type)


class craft_mse_loss(tf.keras.Model):
    def __init__(self):
        super(craft_mse_loss, self).__init__()
        self.confidence_threshold = 0.5

    def call(self, args):
        region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask = args
        """
        temp = np.asarray(region_true)[0]
        img_temp = np.transpose([temp, temp, temp], (1, 2, 0)) * 255
        cv2.imwrite("./logs/temp.jpg", img_temp)
        """
        confidence_mask = tf.greater_equal(confidence, tf.constant(self.confidence_threshold, dtype=tf.float32))
        confidence = tf.where(confidence_mask, confidence, tf.zeros_like(confidence))

        l_region = tf.pow(region_true - region_pred, 2)
        l_region = l_region * confidence

        l_affinity = tf.pow(affinity_true - affinity_pred, 2)
        l_affinity = l_affinity * confidence

        l_total = l_region + l_affinity
        hard_bg_mask = batch_ohem(l_total, fg_mask, bg_mask)
        train_mask = hard_bg_mask + fg_mask
        l_total = l_total * train_mask

        # show loss map
        """
        save_log(region_pred[0], l_region[0], region_true[0], fg_mask[0], hard_bg_mask[0], "region", "0")
        save_log(affinity_pred[0], l_affinity[0], affinity_true[0], fg_mask[0], hard_bg_mask[0], "affinity", "0")
        # """

        loss = tf.reduce_sum(l_total) / (tf.reduce_sum(confidence * train_mask) + tf.keras.backend.epsilon())
        return loss, l_region, l_affinity, hard_bg_mask

class craft_mae_loss(tf.keras.Model):
    def __init__(self):
        super(craft_mae_loss, self).__init__()
        self.confidence_threshold = 0.5

    def call(self, args):
        region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask = args

        confidence_mask = tf.greater_equal(confidence, tf.constant(self.confidence_threshold, dtype=tf.float32))
        confidence = tf.where(confidence_mask, confidence, tf.zeros_like(confidence))

        l_region = tf.abs(region_true - region_pred)
        l_region = l_region * confidence

        l_affinity = tf.abs(affinity_true - affinity_pred)
        l_affinity = l_affinity * confidence

        l_total = l_region + l_affinity
        hard_bg_mask = batch_ohem(l_total, fg_mask, bg_mask)
        train_mask = hard_bg_mask + fg_mask
        l_total = l_total * train_mask

        return tf.reduce_sum(l_total) / (tf.reduce_sum(confidence * train_mask) + tf.keras.backend.epsilon())


class craft_huber_loss(tf.keras.Model):
    def __init__(self):
        super(craft_huber_loss, self).__init__()
        self.confidence_threshold = 0.5

    def huber_loss(self, y_true, y_pred, threshold=0.5):
        residual = tf.abs(y_true - y_pred)
        large_loss = 0.5 * tf.pow(y_true - y_pred, 2)
        small_loss = threshold * residual - 0.5 * tf.square(threshold)
        return tf.where(tf.less(residual, threshold), large_loss, small_loss)

    def call(self, args):
        region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask = args

        confidence_mask = tf.greater_equal(confidence, tf.constant(self.confidence_threshold, dtype=tf.float32))
        confidence = tf.where(confidence_mask, confidence, tf.zeros_like(confidence))

        l_region = self.huber_loss(region_true, region_pred)
        l_region = l_region * confidence

        l_affinity = self.huber_loss(affinity_true, affinity_pred)
        l_affinity = l_affinity * confidence

        l_total = l_region + l_affinity

        # hard_bg_mask = ohem(l_total, fg_mask, bg_mask)
        hard_bg_mask = batch_ohem(l_total, fg_mask, bg_mask)
        # hard_bg_mask = bg_mask
        train_mask = hard_bg_mask + fg_mask
        l_total = l_total * train_mask

        y = tf.reduce_sum(l_total) / (tf.reduce_sum(confidence * train_mask) + tf.keras.backend.epsilon())
        return y
