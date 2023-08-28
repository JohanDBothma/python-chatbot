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

# Read the training data from the phrases.json file
lemmatizer = WordNetLemmatizer()

phrases = json.loads(open('phrases.json').read())

# Initialize empty lists
words = []
classes = []
documents = []
# Add ignore symbols
ignoreSymbols = ['?', '!', '.', ',']

# Loop through all phrases
for phrase in phrases['phrases']:
	for pattern in phrase['patterns']:
		# Tokenize the pattern to words in a list
		wordList = nltk.word_tokenize(pattern)
		# Add each word from the list to the words list
		words.extend(wordList)
		# Add tuple to documents
		documents.append([wordList, phrase['tag']])
		# If the current tag is not in the classes list, add it
		if phrase['tag'] not in classes:
			classes.append(phrase['tag'])

# Take list of words, lemmatizes and creates a new list that excludes the list of symbols
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreSymbols]

# Sort words alphabetically
words = sorted(set(words))

# Sort the list of classes
classes = sorted(set(classes))

# Create pickles of the words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))