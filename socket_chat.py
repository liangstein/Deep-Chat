import os;
import pickle;
import socket;
import numpy as np;
from _thread import start_new_thread;
from keras.models import load_model;
from keras.preprocessing.text import text_to_word_sequence;
DIR=os.getcwd();maxlen=9;maxlenq=10;
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "0.0.0.0";port = 8002
with open(DIR+"/word_index","rb") as f:
  word_index_chi=pickle.load(f);

with open(DIR+"/index_word","rb") as f:
  index_word_chi=pickle.load(f);

model=load_model(DIR+"/model_v1");
def input_2_vec(input):
    a=np.zeros((1,maxlenq));
    for i,ele in enumerate(list(reversed(text_to_word_sequence(input)))):
        if ele in word_index_chi:a[0,i]=word_index_chi[ele];
        else:a[0,i]=word_index_chi["UNK"];
    return a;

def chat(input):
    input_vec=input_2_vec(input);
    predict=model.predict(input_vec)[0];
    predict_max=[];answer=[];
    for j in predict:
        predict_max.append(list(j).index(max(j)));
    for j in predict_max:
        if j==0:break;
        elif j==word_index_chi["UNK"]:continue;
        else:
            answer.append(index_word_chi[j]);
    a=" ".join(answer)
    return a;

serversocket.bind((host, port))
serversocket.listen(5);
print("Server Started on {} port {}".format(host,str(port)))
def client_thread(client):
    while 1:
        data=client.recv(1024).decode().replace("\r\n","");
        print("Reveived data is: " + data+"\n");
        answer=chat(data);
        client.send(("Bot: "+answer+"\n").encode());
        print("Responce Sent");

while 1:
    client, addr = serversocket.accept();
    print("Got a connection from {} port {}".format(str(addr[0]), str(addr[1])))
    start_new_thread(client_thread,(client,));