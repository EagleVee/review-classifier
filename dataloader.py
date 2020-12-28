from gensim.models import Word2Vec
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
    reviewList = []
    negative_data = pd.read_csv(negative_path, sep="\n", header=None, error_bad_lines=False)
    positive_data = pd.read_csv(positive_path, sep="\n", header=None, error_bad_lines=False)
    neutral_data = pd.read_csv(neutral_path, sep="\n", header=None, error_bad_lines=False)

    for review in negative_data[0]:
        reviewList.append(Review(review, 'negative'))

    for review in positive_data[0]:
        reviewList.append(Review(review, 'positive'))

    for review in neutral_data[0]:
        reviewList.append(Review(review, 'negative'))

    random.shuffle(reviewList)

    reviews = []
    labels = []
    for labeledReview in reviewList:
        reviews.append(labeledReview.review)
        labels.append(labeledReview.label)

    return reviews, labels


reviews, labels = readdata()
input_gensim = []
for review in reviews:
    input_gensim.append(review.split())

model = Word2Vec(input_gensim, size=128, window=5, min_count=0, workers=4, sg=1)
model.wv.save("word.model")
