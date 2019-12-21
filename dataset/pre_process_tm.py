import json
import re
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds


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


def load_conversations(inputs, outputs, dialog_json, max_sample=50000):
    for dialog in dialog_json:
        utterances = dialog['utterances']
        for i in range(len(utterances) - 1):
            input = utterances[i]['text'].replace('\n', '').strip()
            inputs.append(preprocess_sentence(input))
            output = utterances[i + 1]['text'].replace('\n', '').strip()
            outputs.append(preprocess_sentence(output))
            if len(inputs) > max_sample:
                return inputs, outputs
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
            pass

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=max_length, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=max_length, padding='post')

    return tokenized_inputs, tokenized_outputs


def conversations_to_dataset(inputs, outputs, max_length):
    questions, answers = tokenize_and_filter(inputs, outputs, max_length)
    return questions, answers


def taskmaster_dataset(max_length, batch_size, buffer_size, save_tokenizer, max_sample=50000):
    with open('data/self-dialogs.json', 'r') as f:
        self_dialog_json = json.load(f)

    with open('data/woz-dialogs.json', 'r') as f:
        woz_dialog_json = json.load(f)

    inputs, outputs = [], []
    inputs, outputs = load_conversations(inputs, outputs, self_dialog_json, max_sample)
    inputs, outputs = load_conversations(inputs, outputs, woz_dialog_json, max_sample)

    print('Sample question: {}'.format(inputs[45]))
    print('Sample answer: {}'.format(outputs[45]))

    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(inputs + outputs, target_vocab_size=2 ** 13)
    # Define start and end token to indicate the start and end of a sentence
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    if save_tokenizer:
        with open('tm_tokenizer.pkl', 'wb') as f:
            pickle.dump((tokenizer, start_token, end_token), f)

    questions, answers = tokenize_and_filter(inputs, outputs, max_length, tokenizer, start_token, end_token)

    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print(dataset)

    return dataset, tokenizer, start_token, end_token
