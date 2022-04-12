import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import spacy
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchtext.legacy.data import Field, BucketIterator,TabularDataset

try: 
    english=spacy.load('en')
    french=spacy.load('fr')
except:
    english=spacy.load('en_core_web_sm')
    french=spacy.load('fr_core_news_sm')

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


# data preprocessing and saving it to a csv file
path_arr=["train","dev","test"]

for x in path_arr:
    en_path="../data/ted-talks-corpus/{}.en".format(x)
    with open(en_path,"r") as file:
        en_text=file.readlines()
        en_text=[handle_punctuation(text.lower()) for text in en_text]
    fr_path="../data/ted-talks-corpus/{}.fr".format(x)
    with open(fr_path,"r") as file:
        fr_text=file.readlines()
        fr_text=[handle_punctuation(text.lower()) for text in fr_text]
    train=pd.DataFrame({'source':en_text,'target':fr_text}).to_csv("../pkl/"+x+".csv",index=False)

# tokenizer and vocab building
batch_size=16  # throwing memory error for more than 16
def en_tokenizer(text):
    return  [token.text for token in english.tokenizer(text)]

def fr_tokenizer(text):
    return  [token.text for token in french.tokenizer(text)]

en_field = Field(tokenize=en_tokenizer,lower=True,init_token="<start>",eos_token="<stop>")
fr_field = Field(tokenize=fr_tokenizer,lower=True,init_token="<start>",eos_token="<stop>")
train_data,valid_data,test_data=TabularDataset.splits(path="../pkl/",train="train.csv",validation="dev.csv",test="test.csv",format="csv",fields=[('source',en_field),('target',fr_field)],skip_header=True)

en_field.build_vocab(train_data)
fr_field.build_vocab(train_data)

train_iter, valid_iter, test_iter = BucketIterator.splits((train_data, valid_data, test_data), batch_size = batch_size,sort_within_batch=True,sort_key=lambda x: len(x.source),device = device)


# model defining
class Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,num_layers,dropout=0.5):
        super(Encoder,self).__init__()
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.embedding_size=embedding_size
        self.dropout=nn.Dropout(dropout)
        self.embedding=nn.Embedding(self.vocab_size,self.embedding_size)
        self.lstm=nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers,dropout=dropout)

    def forward(self,input_size):
        embedding=self.dropout(self.embedding(input_size))
        out,(ht,ct)=self.lstm(embedding)
        return ht,ct

class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,dropout,output_size):
        super(Decoder,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.lstm=nn.LSTM(embedding_size,hidden_size,1,dropout=dropout)
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,input_data,ht,ct):
        input_data=input_data.unsqueeze(0)
        embedding=self.dropout(self.embedding(input_data))
        out,(ht,ct)=self.lstm(embedding,(ht,ct))
        out=self.fc(out).squeeze(0)
        return out,ht,ct

class Seq2Seq(nn.Module):
    def __init__(self, Encoder_LSTM, Decoder_LSTM):
        super(Seq2Seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM

    def forward(self, source, target, tfr=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(fr_field.vocab.itos)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        ht, ct = self.Encoder_LSTM(source)

        x = target[0]
        for i in range(1, target_len):
            output, ht, ct = self.Decoder_LSTM(x, ht, ct)
            outputs[i] = output
            best_guess = output.argmax(1)
            x = target[i] if random.random() < tfr else best_guess
        return outputs


encoder_lstm=Encoder(len(en_field.vocab.itos),300,512,1,0.5).to(device)
decoder_lstm=Decoder(len(fr_field.vocab.itos),300,512,0.5,len(fr_field.vocab.itos)).to(device)
model=Seq2Seq(encoder_lstm,decoder_lstm).to(device)
optimizer=optim.Adam(model.parameters(),lr=1e-3)
criterion=nn.CrossEntropyLoss(ignore_index=fr_field.vocab.stoi["<pad>"])
# optimizer=optim.RMSprop(model.parameters(),lr=1e-3)

for name, param in model.named_parameters():
    nn.init.uniform_(param.data, -0.1, 0.1)

# model training and saving it
epochs=20
best_loss=float('inf')
best_model_state={}

for _ in range(epochs):
    train_loss=0.0
    val_loss=0.0
    # training
    model.train()
    for idx,batch in enumerate(train_iter):
        input_sent=batch.source.to(device)
        target=batch.target.to(device)
        output=model.forward(input_sent,target)
        output=output[1:].reshape(-1,output.shape[2])
        target=target[1:].reshape(-1)
        loss=criterion(output,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        train_loss+=loss.item()

    # validation
    model.eval()
    for idx,bat in enumerate(valid_iter):
        input_sent=bat.source.to(device)
        target=bat.target.to(device)
        output=model(input_sent,target)
        output=output[1:].reshape(-1,output.shape[2])
        target=target[1:].reshape(-1)
        loss=criterion(output,target)
        val_loss+=loss.item()

    # saving the best model
    state={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "rng_state":torch.get_rng_state(),
    }
    if best_loss > val_loss:
        best_loss=val_loss
        best_model_state=state

torch.save(best_model_state,'../pkl/MT-1')


# generating the output
def transalate_per_line(model,text,en_field,fr_field,device,max_len=50):
    model.eval()
    tokens=[token.text.lower() for token in english(text)]
    tokens.insert(0, en_field.init_token)
    tokens.append(en_field.eos_token)
    text_to_indices = [en_field.vocab.stoi[token] for token in tokens]
    text_to_indices = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(text_to_indices)
    outputs = [fr_field.vocab.stoi["<start>"]]

    for _ in range(max_len):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        if output.argmax(1).item() == fr_field.vocab.stoi["<stop>"]:
            break

    translated_sentence = [fr_field.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]


import nltk
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
for x in path_arr:
    df = pd.read_csv("../pkl/"+x+".csv")
    fil=open("2019101053_MT1_"+x+".txt","w")
    val=0
    ff=0
    for ind in df.index:
        en=df['source'][ind]
        fr=df['target'][ind]
        tr=" ".join(transalate_per_line(model,en,en_field,fr_field,device)[:-1])
        hypothesis = fr.split()
        reference =  tr.split()
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
        val+=BLEUscore
        print(tr,BLEUscore,sep="\t",file=fil)
    val=val/len(df)
    print("Average value :",val)

