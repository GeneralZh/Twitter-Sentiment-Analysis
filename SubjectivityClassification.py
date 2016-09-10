__author__ = 'User'

from Tweet import cleanse
from Tweet import micro
from Tweet import getpolarity
from nltk.tag.stanford import StanfordPOSTagger
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
import pandas as pd
import numpy as np
import os
import csv


# ################################################# Set up Tagger #####################################################
java_path = "C:/Program Files/Java/jdk1.8.0_60/bin/java.exe"
os.environ['JAVAHOME'] = java_path

path_to_model = \
    "C:/Users/User/PycharmProjects/untitled/stanford-postagger-2015-04-20/models/english-left3words-distsim.tagger"
path_to_jar = \
    "C:/Users/User/PycharmProjects/untitled/stanford-postagger-2015-04-20/stanford-postagger-3.5.2.jar"
tagger = StanfordPOSTagger(path_to_model, path_to_jar)
# ---------------------------------------------------------------------------------------------------------------------


redundant = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "$", ";", ":", "-", "%", "`"]
X, Y, Xt = [], [], []

stop_word = []

strongpos, strongneg, weakpos, weakneg = [], [], [], []
emjpos, emjneg = [], []

with open('strongpos.txt', 'r') as file:
    for item in file:
        word_tag = item.split()
        strongpos += [[word_tag[0], word_tag[1]]]

with open('strongneg.txt', 'r') as file:
    for item in file:
        word_tag = item.split()
        strongneg += [[word_tag[0], word_tag[1]]]

with open('weakpos.txt', 'r') as file:
    for item in file:
        word_tag = item.split()
        weakpos += [[word_tag[0], word_tag[1]]]

with open('weakneg.txt', 'r') as file:
    for item in file:
        word_tag = item.split()
        weakneg += [[word_tag[0], word_tag[1]]]

with open('neg_emj.txt', 'r') as file:
    for item in file:
        emjneg += [item[:-1]]

with open('pos_emj.txt', 'r') as file:
    for item in file:
        emjpos += [item[:-1]]

with open('Stop Words.txt', 'r') as file:
    for item in file:
        stop_word += [item[:-1]]

sub_emj = emjneg + emjpos


sub_tweet, obj_tweet = [], []
pos_tweet, neg_tweet = [], []
sub_tweet_raw, obj_tweet_raw = [], []
all_words = []
all_sub_words = []

with open('1412778302.txt', 'r') as file:
    for line in file:
        tweet = cleanse(line)
        tweet_word = tweet[0]
        all_words += tweet[0]

        polarity, subjectivity = getpolarity(tweet,
            tagger, strongpos, strongneg, weakpos, weakneg, emjpos, emjneg)

        if subjectivity >= 2:
            print("Subjective: " + line)
            sub_tweet += [tweet_word]
            sub_tweet_raw += [line]
            all_sub_words += tweet_word
        elif subjectivity == 0:
            print("Objective: " + line)
            obj_tweet += [tweet_word]
            obj_tweet_raw += [line]
        else:
            print("Weak polarity: " + line)

        if polarity >= 2:
            pos_tweet += [line]
        elif polarity <= -2:
            neg_tweet += [line]

all_words = sorted(set(all_words))
all_sub_words = sorted(set(all_sub_words))
print("Subjective sentences: " + str(len(sub_tweet)) + "; Objective sentences: " + str(len(obj_tweet)))
print("Positive sentences: " + str(len(pos_tweet)) + "; Negative sentences: " + str(len(neg_tweet)))
# print(all_words)
# print(sub_tweet)
# print(obj_tweet)

with open('sub_example_1.txt', 'w') as file:
    for line in sub_tweet_raw:
        file.write(line)

with open('obj_example_1.txt', 'w') as file:
    for line in obj_tweet_raw:
        file.write(line)

with open('pos_example_1.txt', 'w') as file:
    for line in pos_tweet:
        file.write(line)

with open('neg_example_1.txt', 'w') as file:
    for line in neg_tweet:
        file.write(line)


# ################################   Chi-square Test for Polarity Indicator   ########################################
# ################################        |   with word w  |  without word w  ########################################
# ################################   sub  |       f11      |       f12        ########################################
# ################################   obj  |       f21      |       f22        ########################################

with open('sub_chi_square.csv', 'w') as csvfile:
    fieldnames = ['Word', 'Chi-square', 'In Sub', 'In Obj']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for word in all_words:

        f11, f12, f21, f22 = 0, 0, 0, 0
        chi_square = 0

        for line in sub_tweet:
            if word in line:
                f11 += 1
            else:
                f12 += 1
        for line in obj_tweet:
            if word in line:
                f21 += 1
            else:
                f22 += 1

        f11e = (f11 + f21) * (f11 + f12) / (f11 + f12 + f21 + f22)
        f12e = (f12 + f22) * (f11 + f12) / (f11 + f12 + f21 + f22)
        f21e = (f11 + f21) * (f21 + f22) / (f11 + f12 + f21 + f22)
        f22e = (f12 + f22) * (f21 + f22) / (f11 + f12 + f21 + f22)
        n = f11 + f12 + f21 + f22

        # print(f11e, f12e, f21e, f22e)
        if f11e != 0 and f12e != 0 and f21e != 0 and f22e != 0:
            # chi_square = (f11 - f11e) ** 2 / f11e + (f12 - f12e) ** 2 / f12e + \
            #              (f21 - f21e) ** 2 / f21e + (f22 - f22e) ** 2 / f22e
            chi_square = (abs(f11 - f11e) - 0.5) ** 2 / f11e + (abs(f12 - f12e) - 0.5) ** 2 / f12e + \
                         (abs(f21 - f21e) - 0.5) ** 2 / f21e + (abs(f22 - f22e) - 0.5) ** 2 / f22e

        if chi_square > 1:
            writer.writerow({'Word': word, 'Chi-square': chi_square, 'In Sub': f11, 'In Obj': f21})

unigram, bigram = [], []
training_data, testing_data = [], []

# Objective data
with open('obj_example_1.txt', 'r') as file:
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

# Subjective data
with open('sub_example_1.txt', 'r') as file:
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


# Testing data
with open('Testing.txt', 'r') as file:
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
unique_bigram = sorted(set(bigram))
# print(unique_bigram)
# print(unique_unigram)
features = unique_unigram + sub_emj + unique_bigram

print("Removing words with few occurrences...")
temp = []
for item in unique_unigram:
    if unigram.count(item) > 1:  # and len(item) > 1 :
        temp += [item]
unique_unigram = temp

temp = []
for item in unique_bigram:
    if bigram.count(item) > 1:  # and len(item[0]) >=2 and len(item[1]):
        temp += [item]
unique_bigram = temp

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
# model.add(Dropout(dp))

for i in range(N_LAYERS):
    model.add(Dense(N_HN))
    model.add(PReLU())
    model.add(BatchNormalization())
    # model.add(Dropout(dp))

model.add(Dense(output_dim))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Fitting model...")

model.fit(np.array(X_train), np.array(Y_train), nb_epoch=N_EPOCHS)

predDF = model.predict_proba(np.array(X_test))

with open('prediction.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    for line in predDF:
        writer.writerow(line)
