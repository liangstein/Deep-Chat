## Deep-Chat
Deep-Chat is a chatbot implemented by deep neural network. It can understand human language and give a proper responce based on your input. Right now only English is supported.

## Dependency
* Python3(numpy, scipy, pickle, h5py),
* Keras2,
* Tensorflow or Theano (CPU or GPU version)
* Cuda8.0, Cudnn6.0 (If GPU is used)

## Neural Network Implementation
The dataset used is OpenSubtitiles2016. To avoid too much training time on a single GTX1080, only about 600000 dialogues are choosen. The network has the encoder--decoder structure. Either encoder or decoder has two LSTM layers with each layer has 720 hidden dimensions. 

Before fedding data into encoder, the sequence order is reversed and each word is embedded into a vector that has the dimension of 720. The first LSTM layer accepts input by time sequence, the second one accept the input from the first layer and return a total vector which contains all the information of the input sequence. The third layer and the fourth layer are functioned as decoder. The decoder accepts repeated input from the second layer, then return vectors by time sequence. The last dense layer outputs a one-hot vector by time sequence, which forms a responce sentence. 

## Training the network
Even though only 600000 dialogues are used, it took about 145 hours to train the network. On a single GTX1080 with a batch size of 800, it can process about 925 dialogues per second. What's more, the sequence limit for input and output are ten words and eight words to avoid too long training. The neural network is designed on Keras, it's trained on the Tensorflow compiled with cuda8 and cudnn6. The loss function used is Categorical-crossentropy and it is reduced to 1.2826, meanwhile the accuracy reaches 76.57%. 

## Use the network and begin chatting
Using the network is easy, just paste the script into a python shell and use chat() function to begin chatting. The necessary files include the python script, the words dictionary and the model checkpoint file. [Here](https://www.dropbox.com/s/d6ywqa2qusuvmtz/model_v1?dl=0) to download model file. Also you can execute "python3 socket_chat.py" in terminal to start a tcp server. After the server is started, you can use a client such as telnet into the server and type directly to start chatting. 
* chat.py - python script file
* socket_chat.py - socket server python script file
* word_index - pickle dump of words to indexes dictionary
* index_word - pickle dump of indexes to words dictionary
* model_v1 - the HDF5 file that contains all the weights of the model

Now let's see some examples:
```
>>> chat("good morning")
'good morning'
>>> chat("hello there")
'hello'
>>> chat("good afternoon")
'good afternoon'
>>> chat("good evening ")
'good evening'
>>> chat("good night")
'good night'
>>> chat("how are you")
'fine'
>>> chat("i love you")
'i love you'
>>> chat("do you have feeling")
'no'
>>> chat("do you love your parents")
'tell everything'
>>> chat("do you love your father")
'yes who is your father joe'
>>> chat("do you love your mother")
'yes'
>>> chat("shall we go hiking this weekend")
'yes may like a check dance'
>>> chat("i like living by sea")
'you did'
>>> chat("have you been to hawaii before")
'no but going to roof'
>>> chat("lying on beach enjoying the sunshine")
'yes'
>>> chat("lying on the beach enjoying the sunshine")
'now'
```


## Copyrights
You can use the model and the word dictionarys when building your own bot. But the author (who is me) should be acknowledged. 
