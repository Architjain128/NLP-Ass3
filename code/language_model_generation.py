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

def prob_per_corpus(model,text,index_dict):
    text=handle_punctuation(text.lower()).split("\n")
    for x in text:
        cc=prob_per_line(x,model,index_dict)
        print(x,cc)

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

def prob_per_line(line,model,index_dict):
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
    # print(inpdata)
    # print(outpdata)
    for i in range(len(inpdata)):
        print(inpdata[i])
        p=model.predict(inpdata[i])[0][outpdata[i]]
        ans*=(p)
    return ans

if Mode=="train":
    filename = "./data/europarl-corpus/train.europarl"
    with open(filename, 'r') as f:
        text = f.read()
    text=handle_punctuation(text.lower()).split("\n")

    freq_dict = dict()
    for line in text:
        for word in line.split():
            if word not in freq_dict:
                freq_dict[word] = 1
            else:
                freq_dict[word]+= 1
    freq_dict["<start>"]=len(text)
    freq_dict["<stop>"]=len(text)
    freq_dict["<unk>"]=0

    kill_less = 5
    remove_keys = set()
    for key, value in freq_dict.items():
        if value<kill_less:
            freq_dict["<unk>"]+=value
            remove_keys.add(key)

    for key in remove_keys:
        resp = freq_dict.pop(key, None)
        if resp is None:
            print("error")
    
    counter = 4
    index_dict = dict()
    index_dict["<start>"] = 2
    index_dict["<stop>"] = 3
    index_dict["<unk>"] = 1
    index_dict["<pad>"] = 0
    for key in freq_dict.keys():
        if key not in ["<start>", "<stop>", "<unk>","<pad>"]:
            index_dict[key] = counter
            counter+=1

    word_dict = dict()
    for key, value in index_dict.items():
        word_dict[value] = key 
    
    unique_words = len(index_dict)
    
    max_inp_len=0
    total_words=0
    for x in text:
        x=x.split()
        max_inp_len=max(max_inp_len,len(x)+1)
        total_words+=len(x)+2
    
    input_data = []
    output_data = []
    for x in text:
        x=x.split()
        x=refine(x,index_dict)
        for i in range(len(x)-1):
            temp=x[:i+2]
            input_data.append(np.array(temp[:-1]))
            output_data.append(temp[-1:])

    del text
    
    filename = "./data/europarl-corpus/dev.europarl"
    with open(filename, 'r') as f:
        text = f.read()
    text=handle_punctuation(text.lower()).split("\n")
    dev_input_data = []
    dev_output_data = []
    for x in text:
        x=x.split()
        x=refine(x,index_dict)
        for i in range(len(x)-1):
            temp=x[:i+2]
            dev_input_data.append(np.array(temp[:-1]))
            dev_output_data.append(temp[-1:])
    
    del text
    output_data=np.array(output_data)
    dev_output_data=np.array(dev_output_data)
    MAX_LE=20
    
    
    for i in range(len(input_data)):
        arr=[]
        for j in range(max_inp_len-len(input_data[i])):
            arr.append(index_dict["<pad>"])
        for j in range(len(input_data[i])):
            arr.append(input_data[i][j])   
        input_data[i]=arr[-MAX_LE:]

    for i in range(len(dev_input_data)):
        arr=[]
        for j in range(0,MAX_LE-len(dev_input_data[i]),1):
            arr.append(index_dict["<pad>"])
        for j in range(len(dev_input_data[i])):
            arr.append(dev_input_data[i][j])
        dev_input_data[i]=arr[-MAX_LE:]
        
    input_data=np.array(input_data)
    dev_input_data=np.array(dev_input_data)

    model=Sequential()
    model.add(Embedding(input_dim=total_words,output_dim=3,input_length=max_inp_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(200))
    model.add(Dense(len(index_dict), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    model.fit(input_data,to_categorical(output_data,len(index_dict)), epochs=1,batch_size=8,verbose=True,validation_data=(dev_input_data,to_categorical(dev_output_data,len(index_dict))))
    
    model.save("./pkl/language_model")
    pickle.dump(index_dict,open("./pkl/q1_index.pkl","wb"))
    pickle.dump(word_dict,open("./pkl/q1_word.pkl","wb"))
    perplexity_per_corpus(model,"./data/europarl-corpus/train.europarl","./pkl/q1train.txt")
    perplexity_per_corpus(model,"./data/europarl-corpus/test.europarl","./pkl/q1test.txt")

else : 
    model = keras.models.load_model("./saved_pkl/language_model_en")
    index_dict = pickle.load(open("./saved_pkl/q1_index.pkl","rb"))
    word_dict = pickle.load(open("./saved_pkl/q1_word.pkl","rb"))
    os.system("clear")
    model.summary()
    max_inp_len=150
    sentence=input("Enter a sentence : ")
    prob_per_corpus(model,sentence,index_dict)