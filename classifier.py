from gensim.models import Word2Vec
import os
import pandas as pd
import random

path = './data/'
negative_path = 'dataset/train/train_negative_tokenized.txt'
positive_path = 'dataset/train/train_positive_tokenized.txt'
neutral_path = 'dataset/train/train_neutral_tokenized.txt'


class Review:
    def __init__(self, review, label):
        self.review = review
        self.label = label

    def get_one_hot_label(self):
        if self.label == 'positive':
            return [0, 1, 0]
        elif self.label == 'negative':
            return [0, 0, 1]
        else:
            return [1, 0, 0]


def readdata():
    train_set = []
    negative_data = pd.read_csv(negative_path, sep="\n", header=None, error_bad_lines=False)
    positive_data = pd.read_csv(positive_path, sep="\n", header=None, error_bad_lines=False)
    neutral_data = pd.read_csv(neutral_path, sep="\n", header=None, error_bad_lines=False)

    for review in negative_data:
        train_set.append(Review(review, 'negative'))

    for review in positive_data:
        train_set.append(Review(review, 'positive'))

    for review in neutral_data:
        train_set.append(Review(review, 'negative'))

    random.shuffle(train_set)

    return train_set


train_set = readdata()
print(train_set)
