import tensorflow as tf
from dataset.pre_process_tm import preprocess_sentence
from model.transformer import transformer


class Predictor:
    def __init__(self, cfg):
        cfg = __import__(cfg, globals(), locals(),
                         ['MAX_LENGTH',
                          'VOCAB_SIZE',
                          'NUM_LAYERS',
                          'UNITS',
                          'HALF_D_MODEL',
                          'NUM_HEADS',
                          'DROP_OUT',
                          'OPTIMIZER',
                          'LOSS_FUNCTION',
                          'EPOCHS',
                          'TOKENIZER',
                          'START_TOKEN',
                          'END_TOKEN',
                          'NAME'], 0)
        self.cfg = cfg

        def accuracy(y_true, y_pred):
            # ensure labels have shape (batch_size, MAX_LENGTH - 1)
            y_true = tf.reshape(y_true, shape=(-1, cfg.MAX_LENGTH - 1))
            acc = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
            return acc

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

        model.compile(optimizer=cfg.OPTIMIZER, loss=loss_function, metrics=[accuracy])

        model.load_weights("./weights/%s_weights.ckpt" % cfg.NAME)
        self.model = model

    def predict(self, sentence):
        cfg = self.cfg
        model = self.model
        prediction = evaluate(sentence, cfg, model)
        predicted_sentence = cfg.TOKENIZER.decode([i for i in prediction if i < cfg.TOKENIZER.vocab_size])
        return predicted_sentence


def evaluate(sentence, cfg, model):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        cfg.START_TOKEN + cfg.TOKENIZER.encode(sentence) + cfg.END_TOKEN, axis=0)

    output = tf.expand_dims(cfg.START_TOKEN, 0)

    for i in range(cfg.MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, cfg.END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)
