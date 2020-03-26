import tensorflow as tf

class RefineNet(tf.keras.Model):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.last_conv = tf.keras.models.Sequential(
            tf.keras.layers.Conv2D(34, 64, kernel_size=3, padding="same"), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(64, 64, kernel_size=3, padding="same"), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(64, 64, kernel_size=3, padding="same"), tf.keras.layers.ReLU(inplace=True)
        )

        self.aspp1 = tf.keras.models.Sequential(
            tf.keras.layers.Conv2D(64, 128, kernel_size=3, dilation_rate=6, padding="same"), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 128, kernel_size=1), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 1, kernel_size=1)
        )

        self.aspp2 = tf.keras.models.Sequential(
            tf.keras.layers.Conv2D(64, 128, kernel_size=3, dilation_rate=12, padding="same"), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 128, kernel_size=1), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 1, kernel_size=1)
        )

        self.aspp3 = tf.keras.models.Sequential(
            tf.keras.layers.Conv2D(64, 128, kernel_size=3, dilation_rate=18, padding="same"), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 128, kernel_size=1), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 1, kernel_size=1)
        )

        self.aspp4 = tf.keras.models.Sequential(
            tf.keras.layers.Conv2D(64, 128, kernel_size=3, dilation_rate=24, padding="same"), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 128, kernel_size=1), tf.keras.layers.ReLU(inplace=True),
            tf.keras.layers.Conv2D(128, 1, kernel_size=1)
        )

    def call(self, x1, x2):
        refine = tf.keras.layers.concatenate([x1, x2], axis=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        out = aspp1 + aspp2 + aspp3 + aspp4
        return out
