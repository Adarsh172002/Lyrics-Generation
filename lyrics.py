import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
import pickle

data = pd.read_csv(r'C:\\Users\\Jitendra\\Downloads\\archive\\taylor_swift_lyrics.csv', encoding='latin1')
data = data[['track_title', 'lyric']]

data['lyric'] = data['lyric'].str.lower()

# Splitting the lyric column into sentences
corpus = []
for lyric in data['lyric']:
    corpus.extend(lyric.split('. '))

# Print the resulting sentences
corpus = list(set(corpus))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
    
    input_sequences.append(n_gram_sequence)

tokenizer.word_index["still"]

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

from tensorflow.keras.utils import to_categorical

# Convert target labels to one-hot encoded format
one_hot_labels = to_categorical(label, num_classes=total_words)

# Save the tokenizer as a pickle file
tokenizer_path = './lyrics/tokenizer.pkl'
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Create and compile the model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words // 2, activation='relu'))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(predictors, one_hot_labels, epochs=100, verbose=1)

def make_lyrics(seed_text, next_words):
    generated_lyrics = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
        generated_lyrics += " " + output_word
    return generated_lyrics


seed_text = "Blue"
next_words = 50

generated_lyrics = make_lyrics(seed_text, next_words)
print(generated_lyrics)
