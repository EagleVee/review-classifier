import gensim.models.keyedvectors as word2vec
from keras.models import load_model
import numpy as np
import re
import string
import pandas as pd

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


model = load_model("models.h5")


def pre_process(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()


# text = "rất ngon"
# text = pre_process(text)
test_data = pd.read_csv('shuffled_test.csv', error_bad_lines=False)
print(test_data)
count = 0
accurate = 0
for index, review in enumerate(test_data.review):
    text = pre_process(review)
    maxtrix_embedding = np.expand_dims(comment_embedding(text), axis=0)
    maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)
    result = model.predict(maxtrix_embedding)
    result = np.argmax(result)
    label = test_data.label[index]
    print(label)
    label_text = ''
    if label == 0:
        label_text = "Neutral"
    elif label == 1:
        label_text = "Positive"
    elif label == -1:
        label_text = "Negative"

    if result == 0:
        print("Label predict: Neutral, True Label: ", label_text)
    elif result == 1:
        print("Label predict: Positive, True label: ", label_text)
    elif result == 2:
        print("Label predict: Negative, True label: ", label_text)

    if result == label or (result == 2 and label == -1):
        accurate = accurate + 1

    count = count + 1

    print("Accuracy: ", accurate / count)

