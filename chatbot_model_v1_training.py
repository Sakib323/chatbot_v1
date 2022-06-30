#!/usr/bin/env python
# coding: utf-8

# In[25]:


import tensorflow
import pandas as pd
import numpy as np
import nltk
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
import os


# In[26]:


#tensorflow import here

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD


# In[27]:


#work/working,play/playing etc are same words lammetizer used to identify them as same word
lemmatizer=WordNetLemmatizer()


# In[28]:


#for textual data it is better to use json instead of csv
intent_data=json.loads(open('chatbot v1 qna.json').read())


# In[29]:


words=[]
classes=[]
document=[]
ignore_letters=['!','?','.',',']


# In[30]:


for intent in intent_data['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        document.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]        


# In[31]:


words=sorted(set(words))
classes=sorted(set(classes))


# In[32]:


#using pickle to save them and use them in future
pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))


# In[33]:


#turning string to numerical value to pass them into tensorflow model
#using one-hot-encoding to turn them into catagorical value
#if the word is available in string(from the user input) then 1 else 0
training=[]
output_empty=[0]*len(classes)
for documents in document:
    bag=[]
    word_patterns=documents[0]
    word_patterns=[lemmatizer.lemmatize(word_data.lower()) for word_data in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_raw=list(output_empty)
    output_raw[classes.index(documents[1])]=1
    training.append([bag , output_raw])


# In[34]:


#creating traing value
random.shuffle(training)
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])


# In[35]:


#creating tensorflow model

model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model_v1.h5',hist)
print("Done")

