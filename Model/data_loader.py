import pandas as pd
import numpy as np
import pickle
import re
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import torch
from torch.utils.data import DataLoader, Dataset , SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Defining the Vocab class to be able to map words to indices and indices to words

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = { 0: "<PAD>"}
        self.n_words = 1  

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word.lower()) 


class Preprocessing():

    def __init__(self, path , target_col , cols_to_include=['id','q1_Title','q1_Body','q1_AcceptedAnswerBody',
                                                        'q1_AnswersBody','q2_Title','q2_Body','q2_AcceptedAnswerBody',
                                                        'q2_AnswersBody','class']):

        self.df = pd.read_csv(path,usecols=cols_to_include)
        f = open('Cleaners.pickle','rb') 
        loaded_obj = pickle.load(f)
        f.close()        
        self.target_col = target_col
        self.punctuation_remove = loaded_obj['punctuations']
        self.misspelt_words = loaded_obj['mispell_dict']
        self.cols_to_include = cols_to_include

    def removing_stop_words_lemmatize(self,text,stop_words=[]):
        filtered_text= []
        lemmatizer = WordNetLemmatizer() 
        sent_list = [sent for sent in sent_tokenize(text)]
        filtered_sentence = []
        for sent in sent_list :
          filtered_sentence.append(' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sent) if word not in stop_words]))
              # print(filtered_sentence)
        filtered_text.append('.'.join(filtered_sentence))
        return filtered_text


    def cleaning_text(self,text,punctuation_remove):
      for punct in self.punctuation_remove:
          text = text.replace(punct, '')

      for wrong,norm in self.misspelt_words.items():
          tobereplaced = '[' + wrong + ']' + '+'
          replace = norm
          re.sub(tobereplaced,replace,text.lower())

      text = re.sub(r"\s\s+" , " ", text)
      text = text.strip()

      return text

    def encoding_target(self,df):
      print("-> Encoding target.../n")
      label_enc = LabelEncoder()
      df[self.target_col] = label_enc.fit_transform(df[self.target_col])

      def replace_encoding(num):
          if num==0: return np.array([1,0,0,0]).astype('int64')
          if num==1: return np.array([0,1,0,0]).astype('int64')
          if num==2: return np.array([0,0,1,0]).astype('int64')
          if num==3: return np.array([0,0,0,1]).astype('int64')

      df[self.target_col] = df[self.target_col].apply(lambda x : replace_encoding(int(x)))
      return df

    def run(self,stop_words = []):
      print("Starting Preprocessing.../n")
      stop_words = stop_words + list(set(stopwords.words('english')))
      for cols in self.cols_to_include :
        if cols not in [self.target_col,'id']:
          filtered_text=[]
          for i in range(len(self.df)):
              text = self.df.iloc[i][cols]
              text = self.cleaning_text(text,self.punctuation_remove)
              text = self.removing_stop_words_lemmatize(text,stop_words)
              filtered_text.append(text[0])
          self.df[cols] = filtered_text
          print("-->Done for - ", cols)
      self.df = self.encoding_target(self.df)

      self.df['answer_text1'] = self.df['q1_AcceptedAnswerBody'] + self.df['q1_AnswersBody']
      self.df['answer_text2'] = self.df['q2_AcceptedAnswerBody'] + self.df['q2_AnswersBody']
      self.df = self.df.drop(['q1_AcceptedAnswerBody','q2_AcceptedAnswerBody','q1_AnswersBody','q2_AnswersBody'],axis=1)

      self.df = self.df[["id","q1_Title","q2_Title","q1_Body","q2_Body","answer_text1","answer_text2","class"]]
      self.new_rest_cols = ["id","q1_Title","q2_Title","q1_Body","q2_Body","answer_text1","answer_text2"]
      print("Done!/n")
      return self.df,self.new_rest_cols


class Bilstm_Dataset(Dataset):

    def __init__(self,df,col,target_col):

        # rest_col = [col for col in rest_col if col not in ['id']]
        self.feats = torch.tensor(list(df[col].values.tolist()) , dtype=torch.long)
        self.target = torch.tensor(list(df[target_col].values.tolist()) , dtype=torch.long)
        self.nsamples = len(df)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        return self.feats[index],self.target[index]

class forming_batches():
  def __init__(self,vocab,trimsize_to_col_dict,df,target,vocab_new=True):
    self.mapping_trimsize = trimsize_to_col_dict
    self.df = df
    self.target = target
    self.vocab = vocab
    
    if vocab_new :
        for column in list(self.df.columns):
          if column == self.target or column =='id': continue
          for item in self.df[column]:
            for i in item.split('.'):
              self.vocab.addSentence(i.lower())
        print("Vocabulary formed ! ")
    else:
        print("Using existing Vocab object! ")
    
  def extracting_indices(self,vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')] #need to remove '.' also 
  
  def trim(self,mapping_trimsize,data):

    for colname,maxlen in mapping_trimsize.items():
      trimmed_res = []
      for i in range(len(data)):
        if len(data[colname].iloc[i]) > maxlen:
          trimmed_res.append(data[colname].iloc[i][0:maxlen])
        
        else:
          diff = maxlen - len(data[colname].iloc[i])
          trimmed_res.append(data[colname].iloc[i] + [0]*diff)
      data[colname] = trimmed_res
    
    return data
  
  def run(self):
    print("Converting words to indices using inverse dictionary...")
    for col in self.df.columns:
      if col in [self.target,'id'] : continue
      result_word2index = []
      for i in range(len(self.df)):
        result=[]
        for sent in self.df[col].iloc[i].split('.'):
          result = result + self.extracting_indices(self.vocab,sent.lower())
        
        result_word2index.append(result)
      self.df[col] = result_word2index
    print("Trimming columns...")
    self.df = self.trim(self.mapping_trimsize,self.df)

    for col in self.df.columns:
      self.df[col] = self.df[col].apply(lambda x : np.array(x).astype('int64').squeeze())

    print("dtype ", type(self.df['q1_Title'].iloc[2]))
    
    return self.df , self.vocab
    
class Config(object):
    embed_size = 300
    hidden_layers = 1
    hidden_size = 128
    bidirectional = True
    output_size = 4
    epochs = 25
    lr = 0.001
    ll_hidden_size=50 #Linear layer hidden sizes
    batch_size = 32
    # max_sen_len = 20 # Sequence length for RNN
    dropout_keep = 0.2
    sample = 1
    split_ratio =0.4
    loss_fn = nn.CrossEntropyLoss()
    patience=25
    delta=0.00
    batch_size_test = None
