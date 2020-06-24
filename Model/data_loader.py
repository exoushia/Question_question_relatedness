import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords')

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
            self.addWord(word) 

class Trim:
    def __init__(self,col_list,max_len_list)

class Preprocessing():

    def __init__(self, path , target_col , cols_to_include=['id','q1_Title','q1_Body','q1_AcceptedAnswerBody',
                                                        'q1_AnswersBody','q2_Title','q2_Body','q2_AcceptedAnswerBody',
                                                        'q2_AnswersBody','class']):

        self.df = pd.read_csv(path,usecols=cols_to_include)
        self.split = train_split_ratio
        with open('Cleaners.pickle') as f:
            loaded_obj = pickle.load(f)        
        self.target_col = target_col
        self.punctuation_remove = loaded_obj['punctuations']
        self.misspelt_words = loaded_obj['mispell_dict']

    def removing_stop_words(self,text,stop_words=[]):
        filtered_text= []
        lemmatizer = WordNetLemmatizer() 
        sent_list = [sent for sent in sent_tokenize(text)]
        filtered_sentence = []
        for sent in sent_list :
          filtered_sentence.append(' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sent) if w not in stop_words]))
              # print(filtered_sentence)
        filtered_text.append('.'.join(filtered_sentence))
        return filtered_text


    def cleaning_text(self,text,punctuation_remove):
        for punct in punctuation_remove:
            text = text.replace(punct, '')

        for wrong,norm in misspelt_words.items():
            tobereplaced = '[' + wrong + ']' + '+'
            replace = norm
            re.sub(tobereplaced,replace,text.lower())

        text = exre.sub(r"\s\s+" , " ", text)
        text = text.strip()

        return text

    def encoding_target(self,df):
        # label_enc = LabelEncoder()
        # df[target_col] = label_enc.fit_transform(df[target_col])
        label_enc = OneHotEncoder()
        df[self.target_col] = label_enc.fit_transform(df[self.target_col])
        return df

    def run(self,df,stop_words = []):
        stop_words = stop_words + set(stopwords.words('english'))
        for cols in cols_to_include if not in [self.target_col,'id']:
            filtered_text=[]
            for i in range(len(df)):
                text = df.iloc[i][cols]
                text = self.cleaning_text(text,self.punctuation_remove)
                text = self.removing_stop_words(text,stopwords)
                filtered_text.append(text)
            df[cols] = filtered_text

        df = self.encoding_target(df)

        df['answer_text1'] = df['q1_AcceptedAnswerBody'] + df['q1_AnswersBody']
        df['answer_text2'] = df['q2_AcceptedAnswerBody'] + df['q2_AnswersBody']
        df = df.drop(['q1_AcceptedAnswerBody','q2_AcceptedAnswerBody','q1_AnswersBody','q2_AnswersBody'],axis=1)


        train , val = 


        return df


class Bilstm_Dataset(Dataset):

    def __init__(self,path,preprocess=False,target_col='',rest_col):
        if(preprocess) :
            preprocess_class = Preprocessing(path,target_col,rest_col+target_col)
            df = preprocess_class.run()
        else:
            df = pd.read_csv(path,usecols=rest_col+target_col)


        self.feats = df[:]
        self.target = df[: , [0]]
        self.nsamples = len(df)


    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        return self.feats[index],self.target[index]


np.random.seed(777)   # for reproducibility
df = pd.read_csv()

dataset = Bilstm_Dataset()
NUM_INSTANCES = dataset.__len__()
NUM_INSTANCES = NUM_INSTANCES*sample
TEST_RATIO = 0.3
TEST_SIZE = int(NUM_INSTANCES * 0.3)

indices = list(range(NUM_INSTANCES))

test_idx = np.random.choice(indices, size = TEST_SIZE, replace = False)
# test_idx = np.array(df['id'].iloc[test_idx])
train_idx = list(set(indices) - set(test_idx))
train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)

train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
test_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = test_sampler)
