# python-chatbot

*All credit goes to https://github.com/abhisarahuja/create_chatbot_using_python*

*https://www.youtube.com/watch?v=t933Gh5fNrc&t=1181s*

# Explaining the model

## Libraries

Below are the libraries we use and what the purpose of each is:

* **import random** - we randomise the data as is best practice with building a model. These reasons include:
	1. Reduce order bias
	2. Improved generalization
	3. Avoid data leakage
	4. Optimal weight updates
	5. Better validation performance
	6. Prevent network "memory"
* **import json** - to read the json file of examples
* **import pickle** - create serializations for use in the model script
* **import numpy as np** - importing numpy for the methods
* **import tensorflow as tf** - importing tensorflow for the methods
* **import nltk** - importing nltk for the methods

## Loading and Preprocessing Data

The chatbot needs to learn from examples of user queries and their corresponding intents(tags). The intents.json file has these examples.

The words in each pattern needs to be preprocessed for consistency and effective learning. We use nltk word_tokenize to tokenize the patterns into individual words, lemmatization to reduce words, and converting to lower case to ensure uniformity. 

