import tensorflow as tf


def init_weights(modules):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)


class vgg16(tf.keras.Model):
    def __init__(self, pretrained=True, freeze=True, input_shape=None):
        super(vgg16, self).__init__()
        if input_shape:
            self.basenet = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape)
        else:
            self.basenet = tf.keras.applications.VGG16()
        pass

        self.slice1 = tf.keras.models.Sequential([])
        self.slice2 = tf.keras.models.Sequential([])
        self.slice3 = tf.keras.models.Sequential([])
        self.slice4 = tf.keras.models.Sequential([])
        for i in range(6):  # h/2, w/2
            self.slice1.add(self.basenet.layers[i])
        for i in range(6, 10):  # h/4, w/4
            self.slice2.add(self.basenet.layers[i])
        for i in range(10, 14):  # h/8, w/8
            self.slice3.add(self.basenet.layers[i])
        for i in range(14, 18):  # h/16, w/16
            self.slice4.add(self.basenet.layers[i])

        self.slice5 = tf.keras.models.Sequential([
            tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding="same"),
            tf.keras.layers.Conv2D(1024, (3, 3), padding="same"),
            tf.keras.layers.Conv2D(1024, (1, 1), padding="same")
        ])
        pass

        if not pretrained:
            None  # TODO
        # init_weights(self.slice5_Conv_1)  # TODO

        if freeze:
            for layer_idx, layer in enumerate(self.net.layers):
                layer.trainable = False
                if layer_idx == 6:  # only first conv
                    break

    def call(self, x):
        self.slice1_out = self.slice1(x)
        self.slice2_out = self.slice2(self.slice1_out)
        self.slice3_out = self.slice3(self.slice2_out)
        self.slice4_out = self.slice4(self.slice3_out)
        self.slice5_out = self.slice5(self.slice4_out)
        return [self.slice5_out, self.slice4_out, self.slice3_out, self.slice2_out, self.slice1_out]
