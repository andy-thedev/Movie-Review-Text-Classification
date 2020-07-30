import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# Only retrieve 88000 most frequently occurring words
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# We can notice that each review is a list of word indices, where each integer represents a word.
# print(train_data[0])

# Normally, we would create our own dictionary of mappings per integer, but we will use the one provided
# By the tensorflow library. The command below gives us tuples of key and integer value
word_index = data.get_word_index()

# Break tuple down into key and integer value
word_index = {k:(v+3) for k, v in word_index.items()}
# We will add padding to reviews with less words to make comparisons of equal length
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Since dataset is just integers, we swap order of dictionary so we can get the word by inputting index values.
# This is due to the nature of the dataset we received. If we utilized our own dataset, it would be desirable
# to switch the format of the tuple to be (integer value, key), so we do not have to do the following:
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Make all data of equal length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                       maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                       maxlen=250)


# Turns array of integers into words
def decode_review(text):
    # if i is not found, we return "?" to avoid crashing
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# Using the above function:
# print(decode_review(test_data[2]))

# MODEL---------------------------------------------------------------------------------------------------------------

# We wish for our program to recognize and understand similar phrases and words, within the right context and use
# ie: Have a great day vs. Have a good day
# So we utilize embedding layers


'''
model = keras.Sequential()
# Create word vectors for each possible word (88000 cases) in 16 dimensions: # coefficients
model.add(keras.layers.Embedding(88000, 16))
# We wish to scale the dimension down, since 16 dimensions is very high, and is a lot of data
# By scaling down, it is easier to compute and train
model.add(keras.layers.GlobalAveragePooling1D())
# The scaled down embedding layer of word vectors are passed to a layer of 16 neurons. Classification occurs here
model.add(keras.layers.Dense(16, activation="relu"))
# The 16 neurons then connect to one output neuron, which will return a value between 0 and 1 (sigmoid properties)
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# Using the model-----------------------------------------------------------------------------------------------------

# We use binary_crossentropy as our loss method, since the sigmoid function focuses around 0 and 1
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Split training data into two sets: Validation and Training
# "The model is initially fit on a training dataset, and successively, the fitted model is used to predict the
#   responses for the observations in the validation dataset. The validation dataset provides an unbiased
#   evaluation of a model fit on the training dataset while tuning the model's hyperparameters
#   ie: the number of hidden layers in a neural network

# Validation data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Batch size: How many movie reviews we will load in at once (memory management in cycles)
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

# Presentation will be in the from of: [loss, accuracy]
print(results)

# Saving the model
# Parameter form: name_of_model.h5
# .h5 is the extension for a saved model in keras and tensorflow
model.save("model.h5")
'''


def review_encode(s):
    # Since starting index is 1, by line 21
    encoded = [1]
    for word in s:
        # Get rid of capitalization. If word is not in the word_index, it will append unknown, as in line 22.
        # Otherwise, the encoded word will be appended in integer index form
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


model = keras.models.load_model("model.h5")

# We now attempt to predict using data outside of the dataset we initially utilized to train and test the model.
# We will use a 10/10 movie review, of the renown film, 'Lion King', from the official imdb website

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        # We remove commas, periods, brackets, etc. to retrieve clean words, since words will be stored as
        # ie: [Company,] if we do not do so. The .strip command is intended for \n's, and we will split words at spaces
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(")", "").replace(":",
                                                                                "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                       maxlen=250)
        predict = model.predict(encode)
        # print(line)
        # print(encode)
        # The prediction will return a number close to 0: if the review is a negative review
        #                            a number close to 1: if the review is a positive review
        print(predict[0])

# Structure output for visualization
'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
'''