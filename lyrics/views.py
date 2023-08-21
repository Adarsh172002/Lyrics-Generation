from django.shortcuts import render
from .models import GeneratedLyrics
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential

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
model.fit(predictors, one_hot_labels, epochs=1, verbose=1)

def make_lyrics(seed_text, next_words, temperature=1.0):
    generated_lyrics = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
        generated_lyrics += " " + output_word
    return generated_lyrics

def generate_lyrics(request):
    if request.method == 'POST':
        seed_text = request.POST['seed_text']
        next_words = int(request.POST['next_words'])

        generated_lyrics = make_lyrics(seed_text, next_words, temperature=0.7)

        # Save the generated lyrics in the database
        GeneratedLyrics.objects.create(seed_text=seed_text, next_words=next_words, generated_lyrics=generated_lyrics)

        return render(request, 'main.html', {'generated_lyrics': generated_lyrics})

    return render(request, 'main.html')
