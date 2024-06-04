# author_of_the_quixote
This project is an exploration into the basics of Natural Language Processing (NLP).
Inspired by Jorge Luis Borges' story "Pierre Menard, Author of the Quixote," the objective is to train an AI model to rewrite the novel "Don Quixote" from its beginning and attempt to produce the ending word-for-word.

## Background
"Don Quixote," written by Miguel de Cervantes in the early 17th century, is considered one of the most important works of Western literature. 
The novel recounts the adventures of Alonso Quijano, a nobleman who reads so many chivalric romances that he loses his sanity and decides to become a knight-errant named Don Quixote.
Accompanied by his loyal squire, Sancho Panza, Don Quixote embarks on a series of misadventures, often mistaking mundane objects for fantastical elements from his books. 
The novel is renowned for its rich narrative, complex characters, and its pioneering use of realism and metafiction. 
It explores themes such as the conflict between reality and illusion, the nature of heroism, and the power of literature, making it a profound and enduring work that has influenced countless writers and thinkers.

One of those writers and thinkers is Jorge Luis Borges, 

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