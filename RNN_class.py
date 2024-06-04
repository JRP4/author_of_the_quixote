import tensorflow as tf
import numpy as np
import os
import re

def Basic_RNN(hidden_units, vocab_size,embedding_dim,learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(tf.keras.layers.SimpleRNN(hidden_units, activation='tanh',return_sequences=True))
    model.add(tf.keras.layers.Dense(vocab_size))

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimiser, metrics=['accuracy'])
    return model

def Basic_RNN_v2(hidden_units, vocab_size,embedding_dim,dropout_rate=0.0,recurrent_dropout_rate=0.0,learning_rate=2e-3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(tf.keras.layers.SimpleRNN(hidden_units, activation='tanh',return_sequences=True,
                        dropout=dropout_rate,recurrent_dropout=recurrent_dropout_rate))
    model.add(tf.keras.layers.SimpleRNN(hidden_units, activation='tanh',return_sequences=True,
                        dropout=dropout_rate,recurrent_dropout=recurrent_dropout_rate))
    model.add(tf.keras.layers.SimpleRNN(hidden_units, activation='tanh',return_sequences=True,
                        dropout=dropout_rate,recurrent_dropout=recurrent_dropout_rate))
    model.add(tf.keras.layers.Dense(vocab_size))

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,clipvalue=5)
    model.compile(loss=loss, optimizer=optimiser, metrics=['accuracy'])
    return model

def GRU_RNN(hidden_units, vocab_size,embedding_dim,learning_rate=2e-3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(tf.keras.layers.GRU(hidden_units, activation='tanh',return_sequences=True))
    model.add(tf.keras.layersGRU(hidden_units, activation='tanh',return_sequences=True))
    model.add(tf.keras.layers.Dense(vocab_size))

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimiser, metrics=['accuracy'])
    return model

def LSTM_RNN(hidden_units, vocab_size,embedding_dim,learning_rate=2e-3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layersEmbedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(tf.keras.layers.LSTM(hidden_units, activation='tanh',return_sequences=True))
    model.add(tf.keras.layers.LSTM(hidden_units, activation='tanh',return_sequences=True))
    model.add(tf.keras.layers.Dense(vocab_size))

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,clipvalue=5)
    model.compile(loss=loss, optimizer=optimiser, metrics=['accuracy'])
    return model

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

  @tf.function
  def generate_one_step(self, inputs):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()
    
    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits = self.model(input_ids)[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
   
    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars