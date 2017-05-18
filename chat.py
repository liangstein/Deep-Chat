import os;
import pickle;
import numpy as np;
from keras.models import load_model;
from keras.preprocessing.text import text_to_word_sequence;
DIR=os.getcwd();maxlen=9;maxlenq=10;
with open(DIR+"/word_index","rb") as f:
  word_index_chi=pickle.load(f);

with open(DIR+"/index_word","rb") as f:
  index_word_chi=pickle.load(f);

model=load_model("/home/liangstein/Deep-Chat/model_v1");
def input_2_vec(input):
    a=np.zeros((1,maxlenq),dtype=np.uint16);
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
