import tensorflow as tf
import numpy as np
import os
import re

#text downloaded from https://www.gutenberg.org/ebooks/996 and has been maually altered so to remove any project gutenberg writing.
#functions heavily influnced by the TensorFlow tutorial https://www.tensorflow.org/text/tutorials/text_generation

def quixote_data_cleaning():
    text = open('quixote_text.txt').read()
    #print(text[:250])

    #removes text representing placeholders for images.
    while text.find('.jpg')!=-1:
        pos= text.find('.jpg')
        text = text.replace(text[pos-6:pos+25], '')

    #remove line-breaks as they are unneeded and only complicate the dataset 
    #however we must leave in /n/n which represents a new paragraph
    pattern = r'(?<!\n)\n(?!\n)'
    text = re.sub(pattern, ' ', text)
    return text

#For training you'll need a dataset of (input, label) pairs. Where input and label are sequences. At each time step the input is the current character and the label is the next character.

#Here's a function that takes a sequence as input, duplicates, and shifts it to align the input and label for each timestep:
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def text_processing(text,sequence_length,BATCH_SIZE,BUFFER_SIZE):
    # The unique characters in the file
    vocab = sorted(set(text))
    vocab_size=len(vocab)
    #print(f'{vocab_size} unique characters')

    #creates a mapping that assigns all characters in the text an index.
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    #using perhaps a better methods
    vocab_size = len(ids_from_chars.get_vocabulary())

    #transforms the text into an array of unicode characters and then into ids.
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    # The batch method lets you easily convert these individual characters to sequences of the desired size.
    sequences = ids_dataset.batch(sequence_length+1, drop_remainder=True) 
    
    dataset = sequences.map(split_input_target)
    
    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    return dataset,ids_from_chars, chars_from_ids, vocab_size

