import pandas as pd
import numpy as np
import pickle, re, os, sys
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.benchmark = True

sys.path.append(os.path.realpath('..'))
from settings import device


# Defining the Vocab class to be able to map words to indices and indices to words
class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>"}
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


class Preprocessing:

    def __init__(self, path, target_col, cols_to_include=['id', 'q1_Title', 'q1_Body', 'q1_AcceptedAnswerBody',
                                                          'q1_AnswersBody', 'q2_Title', 'q2_Body',
                                                          'q2_AcceptedAnswerBody',
                                                          'q2_AnswersBody', 'class']):

        self.df = pd.read_csv(path, usecols=cols_to_include)
        f = open('Data/Cleaners.pickle', 'rb')
        loaded_obj = pickle.load(f)
        f.close()
        self.target_col = target_col
        self.punctuation_remove = loaded_obj['punctuations']
        self.misspelt_words = loaded_obj['mispell_dict']
        self.cols_to_include = cols_to_include
        self.new_rest_cols = ["id", "q1_Title", "q2_Title", "q1_Body", "q2_Body", "answer_text1", "answer_text2"]

    def removing_stop_words_lemmatize(self, text, stop_words=[]):
        filtered_text = []
        lemmatizer = WordNetLemmatizer()
        sent_list = [sent for sent in sent_tokenize(text)]
        filtered_sentence = []
        for sent in sent_list:
            filtered_sentence.append(
                ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sent) if word not in stop_words]))
        # print(filtered_sentence)
        filtered_text.append('.'.join(filtered_sentence))
        return filtered_text

    def cleaning_text(self, text):
        for punct in self.punctuation_remove:
            text = text.replace(punct, '')

        for wrong, norm in self.misspelt_words.items():
            tobereplaced = '[' + wrong + ']' + '+'
            replace = norm
            re.sub(tobereplaced, replace, text.lower())

        text = re.sub(r"\s\s+", " ", text)
        text = text.strip()

        return text

    def run(self, stop_words=[]):
        print("Starting Preprocessing.../n")
        stop_words = stop_words + list(set(stopwords.words('english')))
        for cols in self.cols_to_include:
            if cols not in [self.target_col, 'id']:
                filtered_text = []
                for i in range(len(self.df)):
                    text = self.df.iloc[i][cols]
                    text = self.cleaning_text(text)
                    text = self.removing_stop_words_lemmatize(text, stop_words)
                    filtered_text.append(text[0])
                self.df[cols] = filtered_text
                print("-->Done for - ", cols)

        self.df['answer_text1'] = self.df['q1_AcceptedAnswerBody'] + self.df['q1_AnswersBody']
        self.df['answer_text2'] = self.df['q2_AcceptedAnswerBody'] + self.df['q2_AnswersBody']
        self.df = self.df.drop(['q1_AcceptedAnswerBody', 'q2_AcceptedAnswerBody', 'q1_AnswersBody', 'q2_AnswersBody'],
                               axis=1)

        self.df = self.df[["id", "q1_Title", "q2_Title", "q1_Body", "q2_Body", "answer_text1", "answer_text2", "class"]]

        print("Done!/n")
        return self.df, self.new_rest_cols


class Bilstm_Dataset(Dataset):

    def __init__(self, df, col, target_col):
        # rest_col = [col for col in rest_col if col not in ['id']]
        self.feats = torch.as_tensor(list(df[col].values.tolist()), dtype=torch.long, device=device)
        self.target = torch.as_tensor(list(df[target_col].values.tolist()), dtype=torch.long, device=device)
        self.nsamples = len(df)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        return self.feats[index], self.target[index]


class forming_batches:
    def __init__(self, vocab, trimsize_to_col_dict, df, target, vocab_new=True):
        self.mapping_trimsize = trimsize_to_col_dict
        self.df = df
        self.target = target
        self.vocab = vocab

        if vocab_new:
            for column in list(self.df.columns):
                if column == self.target or column == 'id': continue
                print(column)
                for item in self.df[column]:
                    if item is not np.nan:
                        for i in item.split('.'):
                            self.vocab.addSentence(i.lower())
            print("Vocabulary formed ! ")
        else:
            print("Using existing Vocab object! ")

    def extracting_indices(self, vocab, sentence):
        return [vocab.word2index[word] for word in sentence.split(' ')]  # need to remove '.' also

    def trim(self, mapping_trimsize, data):

        for colname, maxlen in mapping_trimsize.items():
            trimmed_res = []
            for i in range(len(data)):
                if len(data[colname].iloc[i]) > maxlen:
                    trimmed_res.append(data[colname].iloc[i][0:maxlen])

                else:
                    diff = maxlen - len(data[colname].iloc[i])
                    trimmed_res.append(data[colname].iloc[i] + [0] * diff)
            data[colname] = trimmed_res

        return data

    def run(self):
        print("Converting words to indices using inverse dictionary...")
        for col in self.df.columns:
            if col in [self.target, 'id']: continue
            result_word2index = []
            for i in range(len(self.df)):
                result = []
                if self.df[col].iloc[i] is np.nan:
                    result = result + [0]
                else:
                    for sent in self.df[col].iloc[i].split('.'):
                        result = result + self.extracting_indices(self.vocab, sent.lower())

                result_word2index.append(result)
            self.df[col] = result_word2index
        print("Trimming columns...")
        self.df = self.trim(self.mapping_trimsize, self.df)

        for col in self.df.columns:
            if col in [self.target, 'id']: continue
            self.df[col] = self.df[col].apply(lambda x: np.array(x).astype('int64').squeeze())

        print("dtype ", type(self.df['q1_Title'].iloc[2]))

        return self.df, self.vocab
