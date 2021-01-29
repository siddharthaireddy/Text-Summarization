#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import glob
from string import digits
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import *
from keras.initializers import *
import tensorflow as tf
import time, random
import re
import string

from string import digits
from keras.optimizers import Adam



from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)


# In[2]:


import os
short_files = []
cwd = os.getcwd()

#name of the folder having data set in the directory
path = cwd+'\\short_stories'
os.listdir() 

print(path)

# get data file names
filenames = glob.glob(path + "\\*.story")

for file in filenames:
    with open(file, encoding="utf8") as f1:
        short_files.append(f1.read())


# In[3]:


len(short_files)


# In[4]:


def sent_split(a):
  corrected = str(a)
  corrected = re.sub(r"\b(CNN)\b",r"", corrected)
  corrected = re.sub(r'(["!?;])\1+', r'\1', corrected)
  corrected = re.sub(r'\{2,}', r'...', corrected)
  corrected = re.sub(r"//t",r"\t", corrected)
  corrected = re.sub(r"( )\1+",r"\1", corrected)
  corrected = re.sub(r"(\n)\1+",r"\1", corrected)
  corrected = re.sub(r"(\r)\1+",r"\1", corrected)
  corrected = re.sub(r"(\t)\1+",r"\1", corrected)
  corrected.translate(str.maketrans('', '', string.punctuation))
  corrected = corrected.split("highlight",maxsplit=1)
  return corrected

def sent_clean(b):
  b = b.split("\n")
  abc = [] 
  for j in b:
    tokenizer = RegexpTokenizer(r'\w+')
    j = " ".join(tokenizer.tokenize(j))
    k_1 = []
    if len(j)>1:
      p = word_tokenize(j)
      for i in p:
        if i.lower() not in stop_words:
          k_1.append(i.lower())
    if len(k_1) > 0:
      abc.append(" ".join(k_1))
  return " ".join(abc)


def highlight_clean(b):
  abc = [] 
  
  b = re.sub(r"\bhighlight\b",r"", b)
  b = b.split("\n")
  k_1 = []
  for j in b:
    tokenizer = RegexpTokenizer(r'\w+')
    j = " ".join(tokenizer.tokenize(j))
    k_1 = []
    if len(j)>1:
      p = word_tokenize(j)
      for i in p:
        if i.lower() not in stop_words:
          k_1.append(i.lower())
    if len(k_1) > 0:
      abc.append(" ".join(k_1))
  return " ".join(abc)


# In[5]:


stop_words = set(stopwords.words('english'))


# In[6]:


data_set = pd.DataFrame(columns=['sentence', 'highlight'])
data_set_clean = pd.DataFrame(columns=['cleaned sentence', 'cleaned highlight'])


# In[ ]:


for i in range(len(short_files)):
  data_set.loc[i,'sentence'] = sent_split(short_files[i])[0]
  data_set.loc[i,'highlight'] = sent_split(short_files[i])[1]

for i in range(len(short_files)):
  data_set_clean.loc[i,'cleaned sentence'] = sent_clean(sent_split(short_files[i])[0])
  data_set_clean.loc[i,'cleaned highlight'] = highlight_clean(sent_split(short_files[i])[1])


# In[ ]:


data_set_clean.head()


# In[10]:


data_set_clean.isnull().sum()


# In[11]:


data_set_clean.drop_duplicates(inplace=True)


# In[12]:


# Lowercase all characters
data_set_clean['cleaned sentence'] = data_set_clean['cleaned sentence'].apply(lambda x: x.lower())
data_set_clean['cleaned highlight'] = data_set_clean['cleaned highlight'].apply(lambda x: x.lower())


# In[13]:


# Remove quotes
data_set_clean['cleaned sentence'] =data_set_clean['cleaned sentence'].apply(lambda x: re.sub("'", '', x))
data_set_clean['cleaned highlight'] =data_set_clean['cleaned highlight'].apply(lambda x: re.sub("'", '', x))


# In[14]:


exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
data_set_clean['cleaned sentence'] =data_set_clean['cleaned sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
data_set_clean['cleaned highlight']=data_set_clean['cleaned highlight'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))


# In[15]:


# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
data_set_clean['cleaned sentence']=data_set_clean['cleaned sentence'].apply(lambda x: x.translate(remove_digits))
data_set_clean['cleaned highlight']=data_set_clean['cleaned highlight'].apply(lambda x: x.translate(remove_digits))
data_set_clean['cleaned highlight'] = data_set_clean['cleaned highlight'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
data_set_clean['cleaned highlight']=data_set_clean['cleaned highlight'].apply(lambda x: x.strip())
data_set_clean['cleaned sentence']=data_set_clean['cleaned sentence'].apply(lambda x: x.strip())
data_set_clean['cleaned highlight']=data_set_clean['cleaned highlight'].apply(lambda x: re.sub(" +", " ", x))
data_set_clean['cleaned sentence']=data_set_clean['cleaned sentence'].apply(lambda x: re.sub(" +", " ", x))


# In[16]:


data_set_clean.head()


# In[17]:


for i in range(len(data_set_clean)):
  if len(data_set_clean.iloc[i, 0]) >= 200:
    data_set_clean.iloc[i, 0] = data_set_clean.iloc[i, 0][:200]
  
  if len(data_set_clean.iloc[i, 1]) >= 200:
    data_set_clean.iloc[i, 1] = data_set_clean.iloc[i, 1][:200]


# In[18]:


data_set_clean.head()


# In[19]:


data_set_clean.reset_index(inplace = True)


# In[20]:


data_set_clean.columns


# In[21]:


data_set_clean.drop(labels = ['index'], inplace = True, axis = 1)


# In[22]:


#Hyperparameters
batch_size = 64
latent_dim = 256
num_samples = 5000


# In[23]:


data_set_clean.columns


# In[24]:


#Vectorize the data.
input_texts = []
target_texts = []
input_chars = set()
target_chars = set()


    
for i in range(min(num_samples, len(data_set_clean) - 1)):
    input_text = data_set_clean.loc[i,'cleaned sentence']
    target_text = data_set_clean.loc[i,'cleaned highlight']
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    
    for char in input_text:
        if char not in input_chars:
            input_chars.add(char)
    for char in target_text:
        if char not in target_chars:
            target_chars.add(char)

input_chars = sorted(list(input_chars))
target_chars = sorted(list(target_chars))
num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

#Print size
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[25]:


#Define data for encoder and decoder
input_token_id = dict([(char, i) for i, char in enumerate(input_chars)])
target_token_id = dict([(char, i) for i, char in enumerate(target_chars)])

encoder_in_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

decoder_in_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_in_data[i, t, input_token_id[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_in_data[i, t, target_token_id[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_id[char]] = 1.


# In[32]:


#Define and process the input sequence
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

#Using `encoder_states` set up the decoder as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[33]:


#Final model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# In[34]:


#Model Summary
model.summary()


# In[35]:


#Model data Shape
print("encoder_in_data shape:",encoder_in_data.shape)
print("decoder_in_data shape:",decoder_in_data.shape)
print("decoder_target_data shape:",decoder_target_data.shape)


# In[36]:


#Compiling and training the model
model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001), loss='categorical_crossentropy')

model.fit([encoder_in_data, decoder_in_data], decoder_target_data, batch_size = batch_size, epochs=10, validation_split=0.2)


# In[ ]:


#Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


# In[ ]:


reverse_input_char_index = dict((i, char) for char, i in input_token_id.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_id.items())

#Define Decode Sequence
def decode_sequence(input_seq):
    #Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    #Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    #Get the first character of target sequence with the start character.
    target_seq[0, 0, target_token_id['\t']] = 1.

    #Sampling loop for a batch of sequences
    #(to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        #Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        #Exit condition: either hit max length
        #or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        #Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        #Update states
        states_value = [h, c]

    return decoded_sentence


# In[ ]:


for seq_index in range(20):
    input_seq = encoder_in_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


# In[ ]:





# In[ ]:




