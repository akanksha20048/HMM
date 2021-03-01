import nltk
import numpy as np
import itertools
import codecs
import re
import nltk
import string
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from itertools import chain
import seaborn as sn
import pandas as pd

fp = codecs.open('Brown_train.txt', 'r', encoding = 'utf-8', errors = 'ignore')
text = fp.read()
Splitted_text = []
text = text.split('\n')
for s in text:
    Splitted_text.append(s)
text2 = []
for w in Splitted_text:
    space = w.split()
    new = []
    for y in space:
        new.append(y)
    text2.append(new)
SplittedList = []
for s in text2:
    s.insert(0,('<s>_<s>'))
    s.append(('<e>_<e>'))
    SplittedList.append(s)

arr=np.array(SplittedList)
kfold = KFold(3, True, 1)
fold=0
for train, test in kfold.split(arr):

    fold=fold+1
    training=arr[train]
    testing=arr[test]
    
    tainingword = {}
    emission={}
    tag_bigram = {}
    probability={}
    Words=[]
    Tags=[]
    
    for w in training:
        for x in w:
            space = x.split('_')
            if len(space)>=2:
                
                space[0]=space[0].lower()
                try:
                    tainingword[space[1]][space[0]]+=1

                except:
                    tainingword[space[1]]={space[0]:1}  
    for w in training:
      lst=list(nltk.bigrams(w))
      for lst1,lst2 in lst:
        try:
            tag_bigram[lst1[1]][lst2[1]]+=1
        except:
            tag_bigram[lst1[1]]={lst2[1]:1}         
    
    for w in tag_bigram.keys():
      probability[w]={}
      for x in tag_bigram[w].keys():
        probability[w][x]=tag_bigram[w][x]/sum(tag_bigram[w].values())
    count=0   
    tt = {}
    for w in training:
      for x in w:
        space = x.split('_')
        if len(space)>=2:
            space[0]=space[0].lower()
            try:
              if space[1] not in tt[space[0]]:
                tt[space[0]].append(space[1])
            except:
              temp = []
              temp.append(space[1])
              tt[space[0]] = temp

    for w in tainingword.keys():
        emission[w]={}
        for x in tainingword[w].keys():
            emission[w][x]=tainingword[w][x]/sum(tainingword[w].values())
            
    
                
    for w in testing:
      for x in w:
        space = x.split('_')
        if len(space)>=2:
            space[0]=space[0].lower()
            try:
              if space[1] not in tt[space[0]]:
                tt[space[0]].append(space[1])
            except:
              temp = []
              temp.append(space[1])
              tt[space[0]] = temp
              
    for w in testing:
      temp_word=[]
      temp_tag=[]
      for x in w:
        space = x.split('_')
        if len(space)>=2:
            temp_word.append(space[0].lower())
            temp_tag.append(space[1])
      Words.append(temp_word)
      Tags.append(temp_tag)

    PTags = []               
    for w in range(len(Words)):   
      wrd = Words[w]             
      t1 = {}              
      for x in range(len(wrd)):
        step = wrd[x]
        if x == 1:                
          t1[x] = {}
          tags = tt[step]
          for t in tags:
              t1[x][t] = ['<s>',0.000001]
        if x>1:
          t1[x] = {}
          completed = list(t1[x-1].keys())   
          curr  = tt[step]               

          for t in curr:                             
            list1 = []
            for pt in completed:                         
              try:
                list1.append(t1[x-1][pt][1]*probability[pt][t]*emission[t][step])
              except:
                list1.append(t1[x-1][pt][1]*0.000001)
            i = list1.index(max(list1))
            var1 = completed[i]
            t1[x][t]=[var1,max(list1)]
    
      prediction = []
      count = t1.keys()
      c = max(count)
      for j in range(len(count)):
        curr_c = c - j
        if curr_c== c:
          prediction .append('<e>')
          prediction .append(t1[curr_c]['<e>'][0])
        if curr_c<c:
          prediction .append(t1[curr_c][prediction [len(prediction )-1]][0])
      PTags.append(list(reversed(prediction )))
       
    test_t= list(chain.from_iterable(Tags))
    
    predicted_t= list(chain.from_iterable(PTags))
    
    cm=confusion_matrix(test_t, predicted_t)
    
    
    recall = recall_score(test_t, predicted_t,average='weighted')
    print('Q1.1 recall for the fold ',fold,' is :',recall)
    
    precision = precision_score(test_t, predicted_t,average='weighted')
    print('precision for the fold',fold,' is:',precision)
    
    f1score= f1_score(test_t, predicted_t,average='weighted')
    print('f1score for the fold',fold,' is:',f1score)
    
    print("Q1.2 tag wise scores for the fold",fold,"are:-")
    print(classification_report(test_t, predicted_t))
    
    print("Q1.3 Confusion matrix for the fold",fold,"is: ")
    print(cm)
    
    print("------------------------------------------------------------------")
    
    
        
        
    
    
    
    
                
    
    
     
            
            
            
            
            
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    