## How to run
### File Strcture
```
.
├── code
│   ├── language_model.py
│   ├── language_model_generation.py
│   ├── machine_translation.py
│   ├── machine_translation2.py
│   ├── machine_translation_model_generation1.py
│   └── machine_translation_model_generation2.py
├── data
│   ├── europarl-corpus
│   │   ├── a.txt
│   │   ├── dev.europarl
│   │   ├── test.europarl
│   │   └── train.europarl
│   ├── news-crawl-corpus
│   │   ├── dev.news
│   │   ├── test.news
│   │   └── train.news
│   ├── ted-talk
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── ted-talks-corpus
│       ├── dev.en
│       ├── dev.fr
│       ├── test.en
│       ├── test.fr
│       ├── train.en
│       └── train.fr
├── readme.md
├── readme.pdf
├── report.md
├── report.pdf
├── requirements.txt
└── saved_pkl
    ├── 2019101053_LM_test.txt
    ├── 2019101053_LM_train.txt
    ├── 2019101053_MT1_test.txt
    ├── 2019101053_MT1_train.txt
    ├── 2019101053_MT2_test.txt
    ├── 2019101053_MT2_train.txt
    ├── MT-1
    ├── MT-2
    ├── en.pkl
    ├── fr.pkl
    ├── language_model_en
    │   ├── assets
    │   ├── keras_metadata.pb
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── language_model_fr
    │   └── model_fr
    │       ├── assets
    │       ├── keras_metadata.pb
    │       ├── saved_model.pb
    │       └── variables
    │           ├── variables.data-00000-of-00001
    │           └── variables.index
    ├── lm20
    ├── lm_e20
    ├── q1_index.pkl
    ├── q1_index_fr.pkl
    ├── q1_word.pkl
    └── q1_word_fr.pkl

```
### Prerequisite
+ Install dependencies
    ```
        pip3 install -r requirements.txt
    ```
    if not working use this file https://pastebin.com/7nQGADrF
+ Install spaCy
    ```
        python3 -m spacy download en --quiet
        python3 -m spacy download fr --quiet
    ```
### Neural Languauge Model
```
    python3 languauge_model.py ../model_path
```

### Machine Translation Model
```
    python3 machine_translation.py ../model_path
```

## Files
### Building models
+ `languauge_model.py`
+ `machine_translation.py`

### Running models
+ `languauge_model_generation.py`
+ `machine_translation_model_generation1.py`
+ `machine_translation_model_generation2.py`


## Miscellaneous
For more you can visit here 
+ Github: https://github.com/Architjain128/NLP-Ass3
+ Models: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/archit_jain_students_iiit_ac_in/EsIJc4PeqhlBn9n-NomW39IBMfr-ZwMfCecbGRdX5PR-LQ?e=WP0dCy 


