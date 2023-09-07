# python-chatbot

*All credit goes to https://github.com/abhisarahuja/create_chatbot_using_python*

*https://www.youtube.com/watch?v=t933Gh5fNrc&t=1181s*

# Training.py - Explaining the model

### Libraries

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

### Loading and Preprocessing Data

The chatbot needs to learn from examples of user queries and their corresponding intents(tags). The intents.json file has these examples.

The words in each pattern needs to be preprocessed for consistency and effective learning. We use nltk word_tokenize to tokenize the patterns into individual words, lemmatization to reduce words, and converting to lower case to ensure uniformity. 

### Building Vocabulary and Labels

To train a model, the chatbot needs to understand the words used in the patterns and their corresponding intents (tags). This is why unique words are collected into *words* and unique tags are collected into *classes*.

*Documents* stores patterns with their corresponding tags, forming a basis for building the training dataset. 

### Creating Bag-of-words Representation

Since a MLM requires numerical input, we need to convert the text pattern to a format that it will understand, thus we use the bag-of-words representation. 

In the bag-of-words representation, we set each word in the vocabulary to be a feature. A feature vector is created for each pattern, where each entry indicates whether the corresponding word appears in the pattern or not, represented by a bool.

### Preparing Training Data and Labels

We have input features *train_x* and corresponding output labels *train_y*.

*train_x* is the bag-of-words representation to allow the model to learn the relationships between words and inputs.

*train_y* is the one-hot encoded labels, where each label has a corresponding class index, set to 1, and the rest are 0. This helps the model understand which intent corresponds to each pattern.

### Defining the Neural Network Model

We use TensorFlow's Keras API to create the neural network. The model consists of input, hidden and output layers:
	1. The input layer size matches the length of the bag-of-words representation.
	2. The hidden layer allows the model to learn the intermediate representations.
	3. The output layer has many neurons as there are unique classes, and it uses softmax activation to provide a probability distribution over the classes.

Dropout layers help mitigate overfitting by randomly dropping a fraction of neurons during training.

### Compiling the model

We need to compile the model, which is done by specifying the loss function, optimizer and evaluation metrics. 'categorical_crossentropy' is the chosen loss function as it handles multi-class classification problems. *Stochastic Gradient Descent* (sgd) is an optimization algorithm that updates the model's weights based on the gradien of the loss function. Accuracy is the metric used.

### Traing the model

We train the model by using the *fit()* method. We provide the training data (*train_x*) and labels (*train_y*), along with the training epochs and batch size. The model will learn to adjust its weights to minimize the loss function. The optimization process involves backpropagation and gradient descent.

### Saving the model

Once the model has been trained, we save it to avoid retraining.
