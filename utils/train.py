import tensorflow as tf
from model.transformer import transformer


def train(cfg):
    cfg = __import__(cfg, globals(), locals(),
                     ['MAX_LENGTH',
                      'DATASET',
                      'BATCH_SIZE',
                      'VOCAB_SIZE',
                      'NUM_LAYERS',
                      'UNITS',
                      'HALF_D_MODEL',
                      'NUM_HEADS',
                      'DROP_OUT',
                      'OPTIMIZER',
                      'LOSS_FUNCTION',
                      'EPOCHS',
                      'NAME'], 0)

    def accuracy(y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, cfg.MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, cfg.MAX_LENGTH - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    tf.keras.backend.clear_session()

    model = transformer(
        vocab_size=cfg.VOCAB_SIZE,
        num_layers=cfg.NUM_LAYERS,
        units=cfg.UNITS,
        d_model=cfg.HALF_D_MODEL * 2,
        num_heads=cfg.NUM_HEADS,
        dropout=cfg.DROPOUT)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("./weights/%s_weights.ckpt" % cfg.NAME,
                                                          verbose=1,
                                                          save_weights_only=True)

    model.compile(optimizer=cfg.OPTIMIZER,
                  loss=loss_function,
                  metrics=[accuracy])
    model.fit(cfg.DATASET, epochs=cfg.EPOCHS, callbacks=[model_checkpoint])
