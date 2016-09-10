__author__ = 'User'

import nltk
from nltk.tag.stanford import StanfordPOSTagger
import os

def cleanse(tweet):  # Return tweet content and emoji

    # ########################################### List of Emoticons #################################################

    happy = [':)', ':D', ':]', ':>', '=)', '=D', '=3', '<3', ':p', ':d', ':-)', ':-D', '(:', '[:',
             '<:', '(=', '(-:']
    sad = [':(', ':\\', ':/', ':<', '=\\', '=/', ':-(', ':\'(', ':c', '):', '/:', '\\:', '>:', '/=',
           '\\=', ')-:', ')\':', '=(']

    # ---------------------------------------------------------------------------------------------------------------

    # ########################################### Simple Cleansing ##################################################

    emj = ''

    all_cap, lengthening = 0, 0

    for i in range(0, len(tweet) - 2):
        if tweet[i].isalpha() and tweet[i + 1].isalpha() and tweet[i + 2].isalpha() \
                and tweet[i] == tweet[i + 1] and tweet[i + 1] == tweet[i + 2]:
            lengthening = 1

    for i in range(0, len(tweet) - 3):
        if tweet[i].isupper() and tweet[i + 1].isupper() and tweet[i + 2].isupper() and tweet[i + 3].isupper():
            all_cap = 1

    temp = ''
    for i in range(0, len(tweet) - 1):
        if tweet[i] == 'I' and tweet[i + 1] == ' ':
            temp += tweet[i]
        else:
            temp += tweet[i].lower()
    temp += tweet[len(tweet) - 1].lower()
    tweet = temp
    # print(tweet)

    tweet = tweet.replace('&amp;', ' and ')
    tweet = tweet.replace('&lt;', '<')
    tweet = tweet.replace('&gt;', ' ')
    tweet = tweet.replace('\\xe2\\x80\\x99', '\'')
    tweet = tweet.replace('\\xe2\\x80\\x9c', ' ')
    tweet = tweet.replace('\\xe2\\x80\\x9d', ' . ')
    tweet = tweet.replace('\\n', ' ; ')

    # Separate \
    temp = ''
    words = tweet.split()
    # print(words)
    for word in words:
        if '\\x' in word and word[0] != '\\':
            temp += word[0:word.index('\\')] + ' ' + word[word.index('\\'):] + ' '
        else:
            temp += word + ' '
    tweet = ' '.join(temp.split())
    # print(tweet)

    # Separate and replace @
    temp = ''
    words = tweet.split()
    for word in words:
        if '@' in word and word[0] != '@':
            if word[-1] != ':':
                temp += word[0:word.index('@')] + ' Bob '
            else:
                temp += word[0:word.index('@')] + ' Bob: '
        elif '@' in word and word[0] == '@':
            if word[-1] != ':':
                temp += ' Bob '
            else:
                temp += ' Bob: '
        else:
            temp += word + ' '
    tweet = ' '.join(temp.split())
    # print(tweet)

    # Separate and remove #
    temp = ''
    words = tweet.split()
    for word in words:
        if '#' in word and word[0] != '#':
            temp += word[0:word.index('#')] + ' ' + word[(word.index('#')+1):] + ' '
        elif '#' in word and word[0] == '#':
            temp += word[1:] + ' '
        else:
            temp += word + ' '
    tweet = ' '.join(temp.split())
    # print(tweet)

    # Replace URL with httpaddr
    temp = ''
    words = tweet.split()
    for word in words:
        if "http:" in word or "https:" in word:
            temp += ' , httpaddr '
            # temp += ' '
        else:
            temp += word + ' '
    tweet = " ".join(temp.split())
    # print(tweet)

    # ---------------------------------------------------------------------------------------------------------------

    # ############################################# Extract Emoji ###################################################

    # Replace emotioncon
    for emotioncon in happy:
        tweet = tweet.replace(emotioncon, ' emj_h ')
    for emotioncon in sad:
        tweet = tweet.replace(emotioncon, ' emj_s ')

    # Add emoji and emotioncon to emj
    temp = ''
    words = tweet.split()
    for word in words:
        if word == 'emj_h' or word == 'emj_s':
            temp += ''
            emj += word + ' '
        elif '\\x' in word:
            # print(word)
            temp += ' . '
            emj += match_emj(word) + ' '
        else:
            temp += word + ' '
    tweet = " ".join(temp.split())
    emj = " ".join(emj.split())

    # ---------------------------------------------------------------------------------------------------------------

    # ######################################### Additional Cleansing ################################################

    tweet = tweet.replace('\\', '')
    tweet = tweet.replace('(', '')
    tweet = tweet.replace(')', '')
    tweet = tweet.replace('*', '')
    tweet = tweet.replace(' u ', ' you ')
    tweet = tweet.replace(' i ', ' I ')
    tweet = tweet.replace(' aint ', ' ain\'t ')
    tweet = tweet.replace(' isnt ', ' isn\'t ')
    tweet = tweet.replace(' werent ', ' weren\'t ')
    tweet = tweet.replace(' doesnt ', ' doesn\'t ')
    tweet = tweet.replace(' dont ', ' don\'t ')
    tweet = tweet.replace(' didnt ', ' didn\'t ')
    tweet = tweet.replace(' arent ', ' aren\'t ')
    tweet = tweet.replace(' shouldnt ', ' shouldn\'t ')
    tweet = tweet.replace(' wouldnt ', ' wouldn\'t ')

    tweet = nltk.word_tokenize(tweet)
    if 'but' not in tweet:
        return tweet, emj, all_cap, lengthening
    else:
        return tweet[tweet.index('but'):], emj, all_cap, lengthening


def match_emj(text):

    emj_list = []

    with open('emj_list.txt', 'r') as myfile:
        for entry in myfile:
            emj_entry = entry.split()
            emj_list += [(emj_entry[0], emj_entry[1])]

    text = text.replace('\\xe2', '\\x00\\xe2')
    text = text.replace('\\xe3', '\\x00\\xe3')
    text = text.replace('\\xef', '\\x00\\xef')

    num = int(len(text)/16)
    # print(text)
    # print(num)

    temp = ''
    for i in range(1, num + 1):
        # print(i)
        emj = text[int(16 * (i - 1)):int(16 * i)]
        # print(emj)
        for item in emj_list:
            if item[0] == emj:
                temp += item[1] + ' '

    text = " ".join(temp.split())

    return text


def label_emj(emj, pos_emj, neg_emj, obj_emj):

    emj_positive, emj_negative, emj_objective = 0, 0, 0
    emjs = emj.split()

    for item in emjs:
        if item in pos_emj:
            emj_positive = 1

    for item in emjs:
        if item in neg_emj:
            emj_negative = 1

    if emj == '':
        emj_objective = 1

    for item in emjs:
        if item in obj_emj:
            emj_objective = 1

    return emj_positive, emj_negative, emj_objective


def label_text(text, pos_word, neg_word, tagger):

    negation = ['n\'t', 'not', 'no', 'nobody', 'none', 'never']

    text_positive, text_negative, text_objective = 0, 0, 1

    text_tag = tagger.tag(text)
    print(text_tag)

    for i in range(len(text_tag)):
        if i == 0:
            word = text_tag[i]
            if word in pos_word:
                text_positive += 1
                text_objective = 0
            if word in neg_word:
                text_negative += 1
                text_objective = 0
        else:
            word = text_tag[i]
            word_previous, tag = text_tag[i - 1]
            if word in pos_word and word_previous not in negation:
                text_positive += 1
                text_objective = 0
            elif word in pos_word and word_previous in negation:
                text_negative += 1
                text_objective = 0
            elif word in neg_word and word_previous not in negation:
                text_negative += 1
                text_objective = 0
            elif word in neg_word and word_previous in negation:
                text_positive += 1
                text_objective = 0

    return text_positive, text_negative, text_objective


def micro(tweet_tag):
    cue = [',', '.', '!', ':', ';','?', 'and', 'because', 'bc', 'coz', 'cuz', 'however', 'but', '...']
    micro_phrase = []

    temp = []
    for word in tweet_tag:
        if word[0] not in cue:
            temp += [word]
        else:
            temp += [word]
            micro_phrase += [temp]
            temp = []
    if temp != []:
        micro_phrase += [temp]

    return micro_phrase


def getpolarity(cleansed_tweet, tagger, strongpos, strongneg, weakpos, weakneg, emjpos, emjneg):

    negation = ['n\'t', 'not', 'no', 'nobody', 'none', 'never']

    tweet_tag = tagger.tag(cleansed_tweet[0])

    tweet_tag_lst = []
    for word in tweet_tag:
        lst = list(word)
        if word[1] == "NN" or word[1] == "NNS" or word[1] == "NNP" or word[1] == "NNPS":
            lst[1] = "noun"
        elif word[1] == "VB" or word[1] == "VBD" or word[1] == "VBZ" \
                or word[1] == "VBN" or word[1] == "VBG" or word[1] == "VBP":
            lst[1] = "verb"
        elif word[1] == "JJ" or word[1] == "JJR" or word[1] == "JJS":
            lst[1] = "adj"
        elif word[1] == "RB" or word[1] == "RBR" or word[1] == "RBS":
            lst[1] = "adverb"
        else: lst[1] = "others"
        tweet_tag_lst += [lst]

    # print(tweet_tag_lst)

    polarity, subjectivity = 0, 0

    micro_phrases = micro(tweet_tag_lst)

    for item in micro_phrases:
        phrase_polarity = 0

        for word in item:
            if word in strongpos:
                phrase_polarity += 2
            elif word in strongneg:
                phrase_polarity -= 2
            elif word in weakpos:
                phrase_polarity += 1
            elif word in weakneg:
                phrase_polarity -= 1

        for word in item:
            if word[0] in negation:
                phrase_polarity *= -1

        # print(item)
        print(item, end='')
        print(" polarity: " + str(phrase_polarity))
        polarity += phrase_polarity
        subjectivity += abs(phrase_polarity)

    emj = cleansed_tweet[1].split()
    emj_polarity = 0
    for item in emj:
        if item in emjpos:
            emj_polarity += 2
            polarity += 2
            subjectivity += 2
        elif item in emjneg:
            emj_polarity -= 2
            polarity -= 2
            subjectivity += 2
    print(emj, end='')
    print(" emoji polarity: " + str(emj_polarity))

    return polarity, subjectivity


# ################################################# Set up Tagger #####################################################
java_path = "C:/Program Files/Java/jdk1.8.0_60/bin/java.exe"
os.environ['JAVAHOME'] = java_path

path_to_model = \
    "C:/Users/User/PycharmProjects/untitled/stanford-postagger-2015-04-20/models/english-left3words-distsim.tagger"
path_to_jar = \
    "C:/Users/User/PycharmProjects/untitled/stanford-postagger-2015-04-20/stanford-postagger-3.5.2.jar"
tagger = StanfordPOSTagger(path_to_model, path_to_jar)
# ---------------------------------------------------------------------------------------------------------------------

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

with open('strongpos.txt', 'r') as file:
    for item in file:
        word_tag = item.split()
        weakneg += [[word_tag[0], word_tag[1]]]


with open('neg_emj.txt', 'r') as file:
    for item in file:
        emjneg += [item[:-1]]

with open('pos_emj.txt', 'r') as file:
    for item in file:
        emjpos += [item[:-1]]
