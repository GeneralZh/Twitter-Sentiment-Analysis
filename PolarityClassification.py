__author__ = 'User'

from Tweet import cleanse
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
import pandas as pd
import numpy as np
import csv


redundant = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "$", ";", ":", "-", "%", "`"]
X, Y, Xt = [], [], []

stop_word = []

unigram, bigram = [], []
training_data, testing_data = [], []
emjpos, emjneg = [], []

with open('Stop Words.txt', 'r') as file:
    for item in file:
        stop_word += [item[:-1]]

with open('neg_emj.txt', 'r') as file:
    for item in file:
        emjneg += [item[:-1]]

with open('pos_emj.txt', 'r') as file:
    for item in file:
        emjpos += [item[:-1]]

sub_emj = emjneg + emjpos

# Positive data
with open('pos_example_1.txt', 'r') as file:
    for line in file:
        tweet = cleanse(line)
        bigram_this = []
        for i in range(0, len(tweet[0]) - 1):
            bigram_this += [(tweet[0][i], tweet[0][i + 1])]
        bigram += bigram_this
        unigram += tweet[0]
        sentence_compo = tweet[0] + tweet[1].split() + bigram_this

        print(tweet)

        training_data += [[sentence_compo, tweet[2], tweet[3], (1, 0)]]

# Negative data
with open('neg_example_1.txt', 'r') as file:
    for line in file:
        tweet = cleanse(line)
        bigram_this = []
        for i in range(0, len(tweet[0]) - 1):
            bigram_this += [(tweet[0][i], tweet[0][i + 1])]
        bigram += bigram_this
        unigram += tweet[0]
        sentence_compo = tweet[0] + tweet[1].split() + bigram_this

        print(tweet)

        training_data += [[sentence_compo, tweet[2], tweet[3], (0, 1)]]


# Testing data
with open('Testing_sub.txt', 'r') as file:
    for line in file:
        tweet = cleanse(line)

        bigram_this = []
        for i in range(0, len(tweet[0]) - 1):
            bigram_this += [(tweet[0][i], tweet[0][i + 1])]

        sentence_compo = tweet[0] + tweet[1].split() + bigram_this

        print(tweet)

        testing_data += [[sentence_compo, tweet[2], tweet[3]]]

for item in redundant:
    unigram = [x for x in unigram if item not in x]
for item in stop_word:
    unigram = [x for x in unigram if item != x]

for item in redundant:
    bigram = [x for x in bigram if item not in x[0] and item not in x[1]]

unique_unigram = sorted(set(unigram))

print("Removing words with few occurrences...")
temp = []
for item in unique_unigram:
    if unigram.count(item) > 1 and len(item) > 1:
        temp += [item]
unique_unigram = temp

unique_bigram = sorted(set(bigram))

temp = []
for item in unique_bigram:
    if bigram.count(item) > 1:  # and len(item[0]) >=2 and len(item[1]):
        temp += [item]
unique_bigram = temp

# print(unique_bigram)
# print(unique_unigram)
features = unique_unigram + sub_emj + unique_bigram

print("The length of feature is: " + str(len(features)))

print("Vectorizing training data...")

for i in training_data:
    X_i = []
    Y += [[i[3][0], i[3][1]]]
    for item in features:
        if item in i[0]:
            X_i.append(1)
        else:
            X_i.append(0)
    X_i.append(i[1])
    X_i.append(i[2])
    X += [X_i]
X_train = pd.DataFrame(X)
Y_train = pd.DataFrame(Y)

print("Vectorizing testing data...")
for i in testing_data:
    X_i = []
    for item in features:
        if item in i[0]:
            X_i.append(1)
        else:
            X_i.append(0)
    X_i.append(i[1])
    X_i.append(i[2])
    Xt += [X_i]
X_test = pd.DataFrame(Xt)

# print(X_train)
# print(Y_train)
# print(X_test)


print("Building neural net...")
input_dim = X_train.shape[1]
output_dim = 2
N_EPOCHS = 25
N_HN = 256
N_LAYERS = 2
# DP = 0.5


model = Sequential()
model.add(Dense(N_HN, input_shape=(input_dim,)))
model.add(PReLU())
# model.add(Dropout(DP))

for i in range(N_LAYERS):
    model.add(Dense(N_HN))
    model.add(PReLU())
    model.add(BatchNormalization())
    # model.add(Dropout(DP))

model.add(Dense(output_dim))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Fitting model...")

model.fit(np.array(X_train), np.array(Y_train), nb_epoch=N_EPOCHS)

print("Predicting on test data...")
predDF = model.predict_proba(np.array(X_test))

with open('prediction.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    for line in predDF:
        writer.writerow(line)