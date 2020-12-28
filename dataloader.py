from gensim.models import Word2Vec
import pandas as pd
import random
import numpy as np

path = './data/'
negative_path = 'dataset/train/train_negative_tokenized.txt'
positive_path = 'dataset/train/train_positive_tokenized.txt'
neutral_path = 'dataset/train/train_neutral_tokenized.txt'
test_path = 'dataset/test/test_tokenized_ANS.txt'


class Review:
    def __init__(self, review, label):
        self.review = review
        self.label = label


def readdata():
    review_list = []
    negative_data = pd.read_csv(negative_path, sep="\n", header=None, error_bad_lines=False)
    positive_data = pd.read_csv(positive_path, sep="\n", header=None, error_bad_lines=False)
    neutral_data = pd.read_csv(neutral_path, sep="\n", header=None, error_bad_lines=False)

    for review in negative_data[0]:
        review_list.append(Review(review, -1))

    for review in positive_data[0]:
        review_list.append(Review(review, 1))

    for review in neutral_data[0]:
        review_list.append(Review(review, 0))

    random.shuffle(review_list)

    reviews = []
    labels = []
    for labeled_review in review_list:
        reviews.append(labeled_review.review)
        labels.append(labeled_review.label)

    return reviews, labels


def readtest():
    review_list = []
    test_data = pd.read_csv(test_path, sep="\n", header=None, error_bad_lines=False)
    reviews = []
    labels = []
    for index, line in enumerate(test_data[0]):
        if index % 2 == 0:
            reviews.append(line)
        else:
            if line == "NEG":
                labels.append(-1)
            elif line == "POS":
                labels.append(1)
            else:
                labels.append(0)

    return reviews, labels


reviews, labels = readdata()
test_reviews, test_labels = readtest()
pd.DataFrame({'review': reviews, 'label': labels}).to_csv('./shuffled_data.csv', index=False)
pd.DataFrame({'review': test_reviews, 'label': test_labels}).to_csv('./shuffled_test.csv', index=False)
