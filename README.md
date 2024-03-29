# Movie Review Connotation Interpreter

## Credit: https://github.com/techwithtim

A repository containing a movie review text classification model using neural networks.

Language: Python  
Libraries: Tensorflow, keras, numpy  
Dataset: keras.datasets.imdb

## Intro
The model takes a movie review (ie: .txt file), and determines if the movie review is likely to be positive or negative (a good movie experience/bad movie experience), determined by an embedding layer of word vectors, and their respective angles.

## /
**Text_Classification.py:**  
The main algorithm (See section "Design description" below)

**model.h5:**  
A saved model from running the main algorithm with comments on line 52 and line 96 removed (See #8 in section: "Design Description")

**test.txt:**  
A 10/10 movie review of the film "Lion King", from the official imdb website to attempt predicting from sources outside of the training/testing data

## Design description

1) Retrieves the 88000 (reconfigurable) most frequently occurring words out of all training data (movie reviews), and creates dictionary mappings of each word to integer

2) Turns all training and testing data into a list of integers of equal length (with added paddings, and max length), according to the generated dictionary mappings

3) Assigns word vectors for each possible word (88000 cases), in 16 dimensions (ie: 16 coefficients in each vector formula), creating an embedding layer

(The embedding layer, through training adjusts the word vectors into groups of narrow ranges of angles, by examining their context through "learns". This allows the model to determine which words are of similar, and distant in connotation)

4) Scales dimensions down into 1 dimension, utilizing the GlobalAveragePooling1D function, for efficient calculation and proccessing.

5) Scaled down embedding layer of word vectors are passed to a layer of 16 neurons, with ReLU activiation functions for classification

6) The 16 neurons connect to one output neuron, with a sigmoid activation function

7) The output neuron returns a value between 0 and 1, where a value closer to 1 is likely to be a positive review, and a value closer to 0 is likely to be a negative review

8) The model may be saved, and loaded. To train, and save a model, remove the ''' quotations at line 52 and line 96. To simply use a saved model, leave the file unchanged.

9) There is also a review encoder, which takes a stripped line of strings, and turns it into a useable format for the model (list of integers), for external source reviews

# Outcome:

Final epoch validation loss: 0.2944

Final epoch validation accuracy: 0.8872

test loss: 0.3257

test accuracy: 0.873
