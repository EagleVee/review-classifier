import gensim.models.keyedvectors as word2vec
from keras.models import load_model
import numpy as np
import re
import string

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


text = "đồ ăn ở đây ngon"
text = pre_process(text)

maxtrix_embedding = np.expand_dims(comment_embedding(text), axis=0)
maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)

result = model.predict(maxtrix_embedding)
result = np.argmax(result)
if result == 0:
    print("Label predict: Neutral")
elif result == 1:
    print("Label predict: Positive")
elif result == 2:
    print("Label predict: Negative")

