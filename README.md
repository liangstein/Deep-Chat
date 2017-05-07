# Deep-Chat
Deep-Chat is a chatbot implemented by deep neural network. It can understand human language and give a proper responce based on your input. Right now only English is supported.

# Dependencies
Python 3 (numpy, scipy, etc...),\\
Keras 2.02,\\
Tensorflow 1.1. You'd better have a power GPU and install the tensorflow-gpu, otherwise the training time is too long. 

# Training the network
To avoid too much training time, 750 thousands of dialogues are used for training. Since the training dataset is not large enough (compared to thousand thousands used by Google), the hidden size of the network should not be too high. Otherwise it's possible that you'll overfit the dataset.

Firstly I trained the network that has three LSTM layers, one is decoder and the last two are encoder. This network was trained by about 540 thousands of dialogues. After training, the network can already understand human language and give simple responces. The hidden dimension is large (1024) because there's only one layer for encoder. Large hidden dimension results long training time. 
 
After trying for several times, I find that the four-layered LSTM is the best. Before feeding dataset into network, each word is embedded into a vector that has the dimension of 768. The first two LSTM layers are decoder and the last two LSTM layers are encoder. Each LSTM has the hidden dimension of 768 too. 

#Right now the network is being trained...
