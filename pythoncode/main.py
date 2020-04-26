#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
columns=["target","id","time","query","user","tweet"]
df=pd.read_csv('twitter.csv',encoding='latin-1',names=columns)


# In[63]:


k=df.drop(["id","time","query","user"],axis=1)


# In[64]:


p=k.sample(n=20000)


# In[65]:


p.target.value_counts()


# In[66]:


decode = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def sentiment(label):
    return decode[int(label)]

p.target = p.target.apply(lambda x: sentiment(x))


# In[2]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


z=[]
for index,row in p.iterrows():
    row['tweet']=re.sub("@[\w]*","",row['tweet'])
    row['tweet']=re.sub("http:[\S]+","",row['tweet'])
    z.append(row['tweet'])


# In[69]:


p['tweet']=z


# In[70]:


p


# In[71]:


appos = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"
}


# In[72]:


y=[]
for index,row in p.iterrows():
    words = row['tweet'].split()
    modify = [appos[word] if word in appos else word for word in words]
    row['tweet'] = " ".join(modify) 
    y.append(row['tweet'])


# In[73]:


p['tweet']=y


# In[74]:


p['tweet'] = p['tweet'].str.replace("[^a-zA-Z#]", " ")


# In[75]:


p['tweet'] = p['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))


# In[76]:


p


# In[77]:


from  nltk.stem import SnowballStemmer


# In[78]:


stemmer = SnowballStemmer("english")


# In[79]:


def preprocess(text, stem=False):
    tokens = []
    for word in text.split():
        if stem:
            tokens.append(stemmer.stem(word))
        else:
            tokens.append(word)
    return " ".join(tokens)

p.tweet = p.tweet.apply(lambda x: preprocess(x))


# In[80]:


p

#stop words are not removed since words like "not" play huge role in sentiment analysis


# In[81]:


#split the dataset into training set and testing set
import random
def Train_Test(data,sr,train,test):
    for x in range(len(data)):
        if random.random()>sr:
            test.append(data[x])
        else:
            train.append(data[x])



train=[]
test=[]
split_ratio=0.75

data=p.values.tolist()
Train_Test(data,split_ratio,train,test)


# In[82]:


print(len(train),len(test))


# In[83]:


list1=[]
for i in train:
    u=i[1].split(" ")
    list1.append(u)


list2=[]
for i in test:
    v=i[1].split(" ")
    list2.append(v)


# In[84]:


import gensim


# In[85]:


model = gensim.models.Word2Vec(size=30,min_count=1,workers=8)


# In[86]:


model.build_vocab(list1)


# In[87]:


words = model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)


# In[88]:


model.train(list1,total_examples=len(list1),epochs=28)


# In[89]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(list1)
tokenizer.fit_on_texts(list2)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# In[90]:


x_train = pad_sequences(tokenizer.texts_to_sequences(list1), maxlen=200)
x_test = pad_sequences(tokenizer.texts_to_sequences(list2), maxlen=200)


# In[91]:


neutral_upper_limit=0.7
neutral_lower_limit=0.3


# In[92]:


df1=pd.DataFrame(train,columns=['target','tweet'])
df2=pd.DataFrame(test,columns=['target','tweet'])


# In[93]:


labels = df1.target.unique().tolist()
labels.append("NEUTRAL")
labels


# In[94]:


from sklearn.preprocessing import LabelEncoder


# In[95]:


encoder = LabelEncoder()
encoder.fit(df1.target.tolist())

y_train = encoder.transform(df1.target.tolist())
y_test = encoder.transform(df2.target.tolist())


# In[96]:


y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# In[97]:


from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding,LSTM,GRU


# In[117]:


embedding_layer = Embedding(vocab_size,16,input_length=200)


# In[136]:


#embedding leyaer used to transform words into vectors,dropout layer is used to avoid overfitting.Twitter data contains bigger variety as compared to set of documents or review for that matter.Hence there is possibility of overfitting.

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(GRU(40,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# In[137]:


model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])


# In[138]:


history = model.fit(x_train, y_train,batch_size=64,epochs=3)


# In[139]:


score = model.evaluate(x_test, y_test, batch_size=64)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])


# In[128]:


def sentiment(score,neutral=True):
    if(neutral):        
        label = "NEUTRAL"
        if(score <= neutral_lower_limit):
            label = "NEGATIVE"
        elif(score >= neutral_upper_limit):
            label = "POSITIVE"
        return label
    else:
        if(score<0.5):
            return "NEGATIVE"
        else:
            return "POSITIVE"


# In[129]:


def predict(text,neutral=True):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=200)
    score = model.predict([x_test])[0]
    label = sentiment(score,neutral=neutral)
    return {"label": label, "score": float(score)}  


# In[130]:


predict("I love the beautiful music")


# In[133]:


predict("I hate the rain Weather is terrible")


# In[134]:


predict("yes")


# In[135]:


predict("no")


# In[7]:


accuracy=[73.09,75.23,74.97,75.88,76.32]
epoch=[1,2,3,4,5]
plt.title("model performance")
plt.plot(epoch,accuracy)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(60,80)
plt.show()


# In[ ]:




