
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, GlobalMaxPool1D
from keras.models import Model

# loading data
base_path = "./toxic/"
input_path = "./toxic/input/"

train_df = pd.read_csv(input_path + "train.csv")
test_df = pd.read_csv(input_path + "test.csv")
subm_df = pd.read_csv(input_path + "sample_submission.csv")

train_df.isnull().any()
test_df.isnull().any()

classes_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets_train = train_df[classes_list].values
comments_list_train = train_df["comment_text"]
comments_list_test = test_df["comment_text"]

# tokenization of comments and indexing of words
max_feat = 20000 # unique words (rows in embedding vector)

tokenizer = Tokenizer(num_words = max_feat)
tokenizer.fit_on_texts(list(comments_list_train)) # create word indices

tokenized_list_train = tokenizer.texts_to_sequences(comments_list_train)
tokenized_list_test = tokenizer.texts_to_sequences(comments_list_test)

# fixing length of comments by padding
num_words = [len(comment) for comment in tokenized_list_train]

plt.hist(num_words, bins = np.arange(0, 410, 10))
plt.show()

# fixing length of comments by padding
max_len = 200
comments_train = pad_sequences(tokenized_list_train, maxlen = max_len)
comments_test = pad_sequences(tokenized_list_test, maxlen = max_len)

# glove embeddings
embed_size = 50 # word vector size

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype = "float32")

embeddings_index = dict(
        get_coefs(*string.strip().split()) # read glove word vectors into a dictionary
        for string in
        open(input_path + "glove.6B.50d.txt")
)

all_embs = np.stack(embeddings_index.values())
emb_mean = all_embs.mean() # use the same mean and stdev of embeddings
emb_std = all_embs.std() # glove has when generating the random init


word_index = tokenizer.word_index
num_words = min(max_feat, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embed_size)) # random init for words not in glove

for word, index in word_index.items(): # iterate over comment dictionary
    if index >= max_feat:
        continue
    embedding_vector = embeddings_index.get(word) # search for embedding from glove
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector # set value if found
    
# specifying model
inp = Input(shape = (max_len, ))

x = Embedding(max_feat, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation = "relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation = "sigmoid")(x)

model = Model(inputs = inp, outputs = x)

model.compile(
        loss = "binary_crossentropy",
        optimizer = "adam",
        metrics = ["accuracy"]
)

# fitting model
model.fit(
        comments_train,
        targets_train,
        batch_size = 32,
        epochs = 2,
        validation_split = 0.1
)

# predictions
subm_df[classes_list] = model.predict([comments_test], batch_size = 1024)
subm_df.to_csv(base_path + "toxic_subm_glove.csv", index = False)  
