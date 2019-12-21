import tensorflow as tf
from model.learning_schedule import CustomSchedule
import pickle

# Transformer
NUM_LAYERS = 2
HALF_D_MODEL = 128
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# Data
MAX_LENGTH = 100
BATCH_SIZE = 32
with open('tm_tokenizer.pkl', 'rb') as tk:
    TOKENIZER, START_TOKEN, END_TOKEN = pickle.load(tk)
VOCAB_SIZE = TOKENIZER.vocab_size + 2

# Train
LEARNING_RATE = CustomSchedule(2 * HALF_D_MODEL)
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
EPOCHS = 20

# Meta
NAME = 'default'
