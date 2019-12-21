from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import pickle

if not os.path.exists('cornell_movie_dialogs.zip'):
    ZIP_PATH = tf.keras.utils.get_file('cornell_movie_dialogs.zip',
                                       origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
                                       extract=True)
CORPUS_PATH = os.path.join(os.path.dirname(ZIP_PATH), "cornell movie-dialogs corpus")
MOVIE_LINES = os.path.join(CORPUS_PATH, 'movie_lines.txt')
MOVIE_CONVERSATIONS = os.path.join(CORPUS_PATH, 'movie_conversations.txt')


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


def load_conversations():
    # dictionary of line id to text
    id2line = {}
    with open(MOVIE_LINES, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(MOVIE_CONVERSATIONS, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if(len(inputs)>1000):
                return inputs,outputs
    return inputs, outputs


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs, max_length, tokenizer, start_token, end_token):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = start_token + tokenizer.encode(sentence1) + end_token
        sentence2 = start_token + tokenizer.encode(sentence2) + end_token
        # check tokenized sentence max length
        if len(sentence1) <= max_length and len(sentence2) <= max_length:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
        else:
            print('long sentence: {}, {}'.format(len(sentence1), len(sentence2)))

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=max_length, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=max_length, padding='post')

    return tokenized_inputs, tokenized_outputs


def cornell_movie_dialogs_dataset(max_length, batch_size, buffer_size, save_tokenizer):
    questions, answers = load_conversations()

    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2 ** 13)
    # Define start and end token to indicate the start and end of a sentence
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    if save_tokenizer:
        with open('cmd_tokenizer.pkl', 'wb') as f:
            pickle.dump((tokenizer, start_token, end_token), f)

    questions, answers = tokenize_and_filter(questions, answers, max_length, tokenizer, start_token, end_token)

    BATCH_SIZE = 64
    BUFFER_SIZE = 20000

    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    DATASET = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))

    DATASET = DATASET.cache()
    DATASET = DATASET.shuffle(BUFFER_SIZE)
    DATASET = DATASET.batch(BATCH_SIZE)
    DATASET = DATASET.prefetch(tf.data.experimental.AUTOTUNE)
    print(DATASET)
    return DATASET, tokenizer, start_token, end_token
