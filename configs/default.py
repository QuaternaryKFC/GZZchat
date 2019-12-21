import tensorflow as tf
from model.learning_schedule import CustomSchedule
from dataset.pre_process_tm import taskmaster_dataset

# Transformer
NUM_LAYERS = 2
HALF_D_MODEL = 128
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# Data
MAX_LENGTH = 100
BATCH_SIZE = 32
BUFFER_SIZE = 20000
SAVE_TOKENIZER = True
MAX_SAMPLE = 10000000
DATASET, TOKENIZER, START_TOKEN, END_TOKEN = taskmaster_dataset(MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE, SAVE_TOKENIZER,
                                                                MAX_SAMPLE)
VOCAB_SIZE = TOKENIZER.vocab_size + 2

# Train
LEARNING_RATE = CustomSchedule(2 * HALF_D_MODEL)
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
EPOCHS = 20

# Meta
NAME = 'default'
