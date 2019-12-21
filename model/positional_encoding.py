import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encode(position, d_model)

    def positional_encode(self, position, d_model):
        angles = self.get_angle(
            position=tf.expand_dims(tf.range(position, dtype=tf.float32), 1),
            i=tf.expand_dims(tf.range(d_model, dtype=tf.float32), 0),
            d_model=d_model)

        sins = tf.math.sin(angles[:, 0::2])
        coss = tf.math.cos(angles[:, 1::2])

        # d_model must be even to allow this
        pos_enc = tf.reshape(
            tf.concat([tf.expand_dims(sins, -1), tf.expand_dims(coss, -1)], axis=-1),
            [tf.shape(sins)[0], -1])
        return tf.cast(tf.expand_dims(pos_enc, 0), tf.float32)

    def get_angle(self, position, i, d_model):
        return position / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
