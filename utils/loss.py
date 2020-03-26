import tensorflow as tf


def ohem(loss, fg_mask, bg_mask, negative_ratio=3.):
    fg_num = tf.reduce_sum(fg_mask)
    bg_num = tf.reduce_sum(bg_mask)

    neg_num = tf.maximum(tf.cast(fg_num * negative_ratio, dtype=tf.int32), tf.constant(10000, dtype=tf.int32))
    neg_num = tf.minimum(tf.cast(bg_num, dtype=tf.int32), neg_num)

    neg_loss = loss * bg_mask
    vals, _ = tf.nn.top_k(tf.reshape(neg_loss, shape=[-1]), k=neg_num)
    bg_bool_mask = tf.cast(bg_mask, dtype=tf.bool)
    hard_bg_bool_mask = tf.logical_and(bg_bool_mask, tf.greater_equal(neg_loss, vals[-1]))
    hard_bg_mask = tf.cast(hard_bg_bool_mask, dtype=tf.float32)

    return hard_bg_mask


def batch_ohem(loss, fg_mask, bg_mask, negative_ratio=3.):
    return tf.map_fn(lambda x: ohem(x[0], x[1], x[2], negative_ratio), elems=[loss, fg_mask, bg_mask], dtype=tf.float32)


class craft_mse_loss(tf.keras.Model):
    def __init__(self):
        super(craft_mse_loss, self).__init__()

    def call(self, args):
        region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask = args

        confidence = confidence

        l_region = tf.pow(region_true - region_pred, 2)
        l_region = l_region * confidence

        l_affinity = tf.pow(affinity_true - affinity_pred, 2)
        l_affinity = l_affinity * confidence

        l_total = l_region + l_affinity
        hard_bg_mask = batch_ohem(l_total, fg_mask, bg_mask)
        train_mask = hard_bg_mask + fg_mask
        l_total = l_total * train_mask

        return tf.reduce_sum(l_total) / (tf.reduce_sum(confidence * train_mask) + tf.keras.backend.epsilon())


class craft_mae_loss(tf.keras.Model):
    def __init__(self):
        super(craft_mae_loss, self).__init__()

    def call(self, args):
        region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask = args

        confidence = confidence

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

    def huber_loss(self, y_true, y_pred, threshold=0.5):
        residual = tf.abs(y_true - y_pred)
        large_loss = 0.5 * tf.pow(y_true - y_pred, 2)
        small_loss = threshold * residual - 0.5 * tf.square(threshold)
        return tf.where(tf.less(residual, threshold), large_loss, small_loss)

    def call(self, args):
        region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask = args

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
