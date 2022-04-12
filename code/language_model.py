import os
import tqdm
import math
import time
import pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Embedding
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

Mode = "test"

def handle_punctuation(text):
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace("!", " ")
    text = text.replace("?", " ")
    text = text.replace(";", " ")
    text = text.replace(":", " ")
    text = text.replace("-", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = text.replace("*", " ")
    text = text.replace("+", " ")
    text = text.replace("=", " ")
    text = text.replace("#", " ")
    text = text.replace("%", " ")
    text = text.replace("$", " ")
    text = text.replace("@", " ")
    text = text.replace("&", " ")
    text = text.replace("^", " ")
    text = text.replace("|", " ")
    text = text.replace("~", " ")
    text = text.replace("`", " ")
    text = text.replace("'", "")
    text = text.replace("\"", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("  ", " ")
    return text

def refine(x,index_dict,fl=False):
    arr=[]
    for i in x:
        if i in index_dict:
            arr.append(index_dict[i])
        else:
            arr.append(index_dict["<unk>"])
    if fl==False:
        return arr
    return arr,len(arr)*2


def perplexity_per_line(model,line):
    ans=0
    MAX_LE=20
    x=handle_punctuation(line.lower()).split()
    x,y=refine(x,index_dict,True)
    inpdata = []
    outpdata = []
    for i in range(len(x)-1):
        temp=x[:i+2]
        inpdata.append(np.array(temp[:-1]))
        outpdata.append(temp[-1:])
    outpdata=np.array(outpdata)

    for i in range(len(inpdata)):
        arr=[]
        for j in range(max_inp_len-len(inpdata[i])):
            arr.append(index_dict["<pad>"])
        for j in range(len(inpdata[i])):
            arr.append(inpdata[i][j])   
        inpdata[i]=arr[-MAX_LE:]
    inpdata=np.array(inpdata)
    
    for i in range(len(inpdata)):
        p=model.predict(inpdata[i])[0][outpdata[i]]
        ans+=math.log(p)
    if y==0:
        ans=0
    else:
        ans/=y
    return math.exp(-ans)

def perplexity_per_corpus(model,path,output):
    filename = path
    val=0
    with open(filename, 'r') as f:
        text = f.read()
    text=handle_punctuation(text.lower()).split("\n")
    fil=open(output,"w")
    k=0
    for x in text:
        if k%1000==0:
            print(k)
        k+=1
        cc=perplexity_per_line(model,x)
        val+=cc
        print(x,cc,sep=" ",file=fil)
    return val/len(text)


import pickle
reconstructed_model  = keras.models.load_model("../saved_pkl/language_model_en")
index_dict = pickle.load(open("../saved_pkl/q1_index.pkl","rb"))
word_dict = pickle.load(open("../saved_pkl/q1_word.pkl","rb"))

def prob_per_line(model,line,word_dict):
    rt=""
    ans=0
    MAX_LE=20
    max_inp_len=150
    x=handle_punctuation(line.lower()).split()
    x,y=refine(x,index_dict,True)
    inpdata = []
    outpdata = []
    for i in range(len(x)-1):
        temp=x[:i+1]
        inpdata.append(np.array(temp[:-1]))
        outpdata.append(temp[-1:])
    outpdata=np.array(outpdata)

    for i in range(len(inpdata)):
        arr=[]
        for j in range(max_inp_len-len(inpdata[i])):
            arr.append(index_dict["<pad>"])
        for j in range(len(inpdata[i])):
            arr.append(inpdata[i][j])   
        inpdata[i]=arr[-MAX_LE:]
    inpdata=np.array(inpdata)
    

    for i in range(len(inpdata)):
        p=model.predict(inpdata[i])[0][outpdata[i]]
        rt+=word_dict[outpdata[i][0]]
        ans+=math.log(p)
    return math.exp(ans),ans,rt

def prob_per_sent(model,text,word_dict): 
    text=handle_punctuation(text.lower()).split("\n")
    for x in text:
        prob,cc,sent=prob_per_line(model,x,word_dict)
        print("Probabilty: ",prob)
        print("log( product of p_i): ",cc)
        print("Predicted Sentence: ",sent)
        break

text=input("Enter sentence")
prob_per_sent(reconstructed_model,text,word_dict)