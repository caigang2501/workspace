import tensorflow as tf
import numpy as np
import pandas as pd

#load data
df = pd.read_csv('./data/01_raw/twitter.csv',encoding='ISO-8859-1')
text = df['SentimentText'].tolist()
labels = df['Sentiment'].tolist()

# data procecing
vocab_size = 10000
embedding_dim = 100
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(text)

sequences = tokenizer.texts_to_sequences(text)
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# constract model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# train and save model 
split = 0.8
split_idx = int(len(padded) * split)

train_sequences = padded[:split_idx]
train_labels = np.array(labels[:split_idx])

val_sequences = padded[split_idx:]
val_labels = np.array(labels[split_idx:])

num_epochs = 10
history = model.fit(train_sequences, train_labels, epochs=num_epochs, validation_data=(val_sequences, val_labels), verbose=2)
model.save('nlp/tf_sentiment_model.h5')

# load model and use
loaded_model = tf.keras.models.load_model('tf_sentiment_model.h5')

test_text = "This is a test sentence."
test_sequence = tokenizer.texts_to_sequences([test_text])
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
result = loaded_model.predict(test_padded)[0][0]
print(result)