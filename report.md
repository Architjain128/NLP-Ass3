# Languauge Model
## Perplexity Scores
|Text File|File name|Score|
|:---:|:---:|:---:|
|Train file|2019101053_LM_train.txt|215.9887094897356|
|Test file|2019101053_LM_test.txt|58.781720914773736|

+ used keras 
+ preprocessing includes punctuation removal and lower casing and used own built dictionnaires for encodings.
+ most got unknown `<unk>` tags as next word for input test file sentences with higher probabilities hence results in better perplexity than train.

# Machine Translation
## Without finetuning
|Text File|File name|BLEU Score|
|:---:|:---:|:---:|
|Train file|2019101053_MT1_train.txt|0.4374342501768364|
|Test file|2019101053_MT1_test.txt|0.41783094446269087|

## With finetuning
|Text File|File name|BLEU Score|
|:---:|:---:|:---:|
|Train file|2019101053_MT2_train.txt|0.4562983567860796|
|Test file|2019101053_MT2_test.txt|0.4456635608763503|


+ used pytorch 
+ preprocessing includes punctuation removal and lower casing and spacy
+ i found an increase of 4.37% in train data and 6.71% in test data by using transfer learning from model trained in `Languauge Model` part

# Other Information
+ used early stopping techniue and saved the best model for dev dataset 
+ oberved the in Machine Translation model it generally stops around epoch 4-6 and in Neural Languauge Model dev error decrease so stopped it for epoch 20
