#Longformer model

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

#import ChatLearner
import os
import re
import sys
import tensorflow as tf
from os import path
sys.path.append(path.abspath('../chatbot'))

from settings import PROJECT_ROOT
from botpredictor import BotPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from botui import bot_ui, chat_func

#Loading Longformer model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")
model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")


def qaLongformer(question, text):
    encoding = tokenizer(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]

    attention_mask = encoding["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    return answer

#Loading book data

keyData = pd.read_csv('../chapterData.txt')

sleep0, soil1, matter2, music3, kitten4, bridge5, limestone6, magnet7, fire8, motion9, light10, electric11 = ' '*12
dataList = [sleep0, soil1, matter2, music3, kitten4, bridge5, limestone6, magnet7, fire8, motion9, light10, electric11]
catList = ['sleep0', 'soil1', 'matter2', 'music3', 'kitten4', 'bridge5', 'limestone6', 'magnet7', 'fire8', 'motion9', 'light10', 'electric11']
for x in range(12): #Load all chapter/text data
    with open('../data/{}.txt'.format(x),'r') as f:
        dataList[x] = f.read().replace('\n', ' ')

#QA function
nlp = spacy.load('en_core_web_sm')

def qaFunc(ques):
    ques = spellCheck(ques)
    doc = nlp(ques)
    arr=[token.lemma_ for token in doc]

    c=[0]*12
    for x in arr:
        if x.isalnum():
            if not keyData[keyData['key'] == x].empty: # condition satisfies if word exist in the list of keywords(keyData)
                z = keyData[keyData['key']==x]['category'].values
                for i in z:
                    c[i]+=1
    cat = np.argmax(c)
    if c[cat] == 0:
        fr = "Papaya"
        ans = re.sub(r'_nl_|_np_', '\n', predictor.predict(session_id, ques)).strip()
    else:
        fr = "LongFormer"
        ans = qaLongformer(ques, dataList[cat])
    print(fr, ': ', ans, sep='')
    #print('Question -> {}\nAnswer -> {}\nCategory -> {}\n\n'.format(ques,ans,catList[cat]))
    return ans

#Spell check function
spell = SpellChecker()

def spellCheck(s):
    s = s.lower()
    x = word_tokenize(s)
    misspelled = spell.unknown(x)
    for word in misspelled:
        s = s.replace(word,spell.correction(word))
    return s

inp = ''
corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')

with tf.Session() as sess:
    predictor = BotPredictor(sess, corpus_dir=corp_dir, knbase_dir=knbs_dir,
                             result_dir=res_dir, result_file='basic')
    # This command UI has a single chat session only
    session_id = predictor.session_data.add_session()
    print("Enter your question: ")
    while True:
        inp = input("> ")
        if inp=='exit':
            break
        qaFunc(inp)
