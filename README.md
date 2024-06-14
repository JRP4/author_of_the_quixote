# author_of_the_quixote
This project is an exploration into the basics of Natural Language Processing (NLP).
Inspired by Jorge Luis Borges' story "Pierre Menard, Author of the Quixote," the objective is to train an AI model to re-produce the ending of the novel "Don Quixote" word-for-word.
The aim of this project is mostly for me to learn about generative AI and the entire space of text generation models from CNNs (characer neural networks) up to fine-tuning state-of-the-art LLMs.

## Background
"Don Quixote", written by Miguel de Cervantes in the early 17th century, is considered one of the most important works of Western literature. 
The novel recounts the adventures of Alonso Quijano, a nobleman who reads so many chivalric romances that he loses his sanity and decides to become a knight-errant named Don Quixote.
Accompanied by his loyal squire, Sancho Panza, Don Quixote embarks on a series of misadventures, often mistaking mundane objects for fantastical elements from his romance books. 
The novel is renowned for its rich narrative, complex characters, and its pioneering use of realism and metafiction and explores themes such as the conflict between reality and illusion, the nature of heroism, and the power of literature, making it a profound and enduring work that has influenced countless writers and thinkers.

One of those writers and thinkers is Jorge Luis Borges, 

A limiting factor in this anaylsis is computational time. 
The aim is to build as good of a model on my laptop with a single hour of training time.
This makes many approaches infeasible.

## Details on current files

`text_processing.py` is a file containing functions on how to import and clean the dataset and get it into a state that can be inputting into a TensorFlow model.
`RNN_class.py` is a group of TensorFlow models including base RNNs, GRUs and LSTMs.
`cnn_quixote_basic.ipynb` is a Jupyter Notebook that demonstrates the basics of Character Neural Network (CNN) renergation using this dataset (and ML model pipline).
`quixote_text` is the dataset, it was taken from the Guttenburg depository and has been manually altered so that only text written by Cervantes remains.

## To-do list
- Investigate the effects of dropout, it increases the loss but does it make the model more robust.
There is an issue in the model where it fails to generalise and even with ~60% accuracy in the training set cannot even a single word on unseen data.
Another possible attempt is to use curriculum learning (or some other alternative to teacher training), this might involve incoorporaing a third party tool to measure how difficult a sequence is.
- Split the end of the data off and use it as a validation set.
Use this to define a early-stopping procedure that identified overfitting.
- Investigate different optimisers and the corrponding parameters.
- LSTM versus GRU
- Eventually create a word-token version.
For example use modern work embeddings.
- Use fine-tuning