# Import the needed libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
# Uncomment downloading these resources if missing
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Read the training data from the intents.json file
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

# Initialize empty lists
words = []
classes = []
documents = []
# Add ignore symbols
ignoreSymbols = ['?', '!', '.', ',']

# Loop through all intents and building vocabulary
for intent in intents['intents']:
	for pattern in intent['patterns']:
		# Tokenize the pattern to words in a list
		wordList = nltk.word_tokenize(pattern)
		# Add each word from the list to the words list
		words.extend(wordList)
		# Add tuple of world list to documents
		documents.append([wordList, intent['tag']])
		# If the current tag is not in the classes list, add it
		if intent['tag'] not in classes:
			classes.append(intent['tag'])

# Take list of words, lemmatizes and creates a new list that excludes the list of symbols
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreSymbols]

# Sort words alphabetically and remove duplicates
words = sorted(set(words))

# Sort the list of classes and remove duplicates
classes = sorted(set(classes))

# Create pickles of the words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# Create bag-of-words representation for each document
# Loop through the documents which composes out of patterns and their tags to create bag-of-word representation
for document in documents:
	# We take the index of the pattern of words from the current document and lemmatize it to a lower string
	bag = []
	wordPatterns = document[0]
	wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
	# Create a binary feature vector of bag
	# If this word is in wordPatterns, add 1 to the bag, else 0
	# For example, if the word is `Thank` then it would match the pattern of `Thank you` in the tag `thanks`
	for word in words:
		if word in wordPatterns:
			bag.append(1) 
		else:
			bag.append(0)
	# Create a row that is the same length of the outputEmpty list
	outputRow = list(outputEmpty)
	# Find where the current document's tag is in the classes list
	outputRow[classes.index(document[1])] = 1
	training.append(bag + outputRow)

# Randomize the training and create a numpy array from the training data
random.shuffle(training)
training = np.array(training)

# Slice training by only the first len(words) col
trainX = training[:, :len(words)]
# Slice training by starting from the column at index len(words) until the end of the sequence
trainY = training[:, len(words):]

# Create Neural Network Model
# Two hidden layers with dropout for regularization
# Output layer with softmax activation
# Model is compiled with the categorical cross-entropy loss and the stochastic gradient descent optimizer (SGD)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
# Adjust overfitting
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model with the training data and labels. Trained 200 times in a batch size of 1. Print progress and metrics with verbosity
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
# Save trained model
model.save('chatbot_model.h5', hist)

print('Done')