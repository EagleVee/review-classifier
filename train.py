from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import os
import pandas as pd
import random
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing import sequence
from tqdm import tqdm

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
    review_list = []
    negative_data = pd.read_csv(negative_path, sep="\n", header=None, error_bad_lines=False)
    positive_data = pd.read_csv(positive_path, sep="\n", header=None, error_bad_lines=False)
    neutral_data = pd.read_csv(neutral_path, sep="\n", header=None, error_bad_lines=False)

    for review in negative_data[0]:
        review_list.append(Review(review, 'negative'))

    for review in positive_data[0]:
        review_list.append(Review(review, 'positive'))

    for review in neutral_data[0]:
        review_list.append(Review(review, 'negative'))

    random.shuffle(review_list)

    reviews = []
    labels = []
    for labeled_review in review_list:
        reviews.append(labeled_review.review)
        labels.append(labeled_review.label)

    return reviews, labels


reviews, labels = readdata()
input_gensim = []
for review in reviews:
    input_gensim.append(review.split())

model = Word2Vec(input_gensim, size=128, window=5, min_count=0, workers=4, sg=1)
model.wv.save("word.model")

model_embedding = word2vec.KeyedVectors.load('./word.model')

word_labels = []
max_seq = 200
embedding_size = 128

for word in model_embedding.vocab.keys():
    word_labels.append(word)


def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    lencmt = len(words)

    for i in range(max_seq):
        indexword = i % lencmt
        if max_seq - i < lencmt:
            break
        if words[indexword] in word_labels:
            matrix[i] = model_embedding[words[indexword]]
    matrix = np.array(matrix)
    return matrix


train_data = []
label_data = []

for x in tqdm(reviews):
    train_data.append(comment_embedding(x))

train_data = np.array(train_data)

for y in tqdm(labels):
    label_ = np.zeros(3)
    try:
        label_[int(y)] = 1
    except:
        label_[0] = 1
    label_data.append(label_)

sequence_length = 200
embedding_size = 128
num_classes = 3
filter_sizes = 3
num_filters = 150
epochs = 50
batch_size = 30
learning_rate = 0.01
dropout_rate = 0.5

x_train = train_data.reshape(train_data.shape[0], sequence_length, embedding_size, 1).astype('float32')
y_train = np.array(label_data)

# Define model
model = keras.Sequential()
model.add(layers.Convolution2D(num_filters, (filter_sizes, embedding_size),
                               padding='valid',
                               input_shape=(sequence_length, embedding_size, 1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(198, 1)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
# Train model
adam = tf.optimizers.Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
print(model.summary())

model.fit(x=x_train[:7000], y=y_train[:7000], batch_size=batch_size, verbose=1, epochs=epochs,
          validation_data=(x_train[:3000], y_train[:3000]))

model.save('models.h5')
