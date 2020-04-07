import tensorflow as tf
from basenet.vgg16 import vgg16


class double_conv(tf.keras.Model):
    def __init__(self, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.Conv_1 = tf.keras.layers.Conv2D(mid_ch, (1, 1))
        self.Relu_1 = tf.keras.layers.ReLU()
        self.Conv_2 = tf.keras.layers.Conv2D(out_ch, (3, 3), padding="same")
        self.Relu_2 = tf.keras.layers.ReLU()

    def call(self, x):
        h = self.Conv_1(x)
        h = self.Relu_1(h)
        h = self.Conv_2(h)
        y = self.Relu_2(h)
        return y


class CRAFT(tf.keras.Model):
    def __init__(self, pretrained=True, freeze=False, input_shape=None):
        super(CRAFT, self).__init__()
        num_class = 2

        # basenet
        self.basenet = vgg16(pretrained, freeze, input_shape=input_shape)

        # U net
        self.UpConv_1 = double_conv(512, 256)
        self.UpConv_2 = double_conv(256, 128)
        self.UpConv_3 = double_conv(128, 64)

        self.UpConv_4 = double_conv(64, 32)

        # conv_cls
        self.conv_cls = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, (1, 1)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(num_class, (1, 1))
        ])
        pass

    def call(self, x):
        h = self.basenet(x)
        y = tf.concat([h[0], h[1]], axis=3)
        y = self.UpConv_1(y)

        y = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(y)
        y = tf.concat([y, h[2]], axis=3)
        y = self.UpConv_2(y)

        y = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(y)
        y = tf.concat([y, h[3]], axis=3)
        y = self.UpConv_3(y)

        y = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(y)
        y = tf.concat([y, h[4]], axis=3)
        feature = self.UpConv_4(y)

        y = self.conv_cls(feature)
        return y, feature
