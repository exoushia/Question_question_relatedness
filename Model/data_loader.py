import torch
from torch.utils.data import DataLoader, Dataset

class Preprocessing():

    def __init__(self, path , target_col ,sample_size,train_split_ratio=None, cols_to_include=['q1_Title','q1_Body','q1_AcceptedAnswerBody',
                                                        'q1_AnswersBody','q2_Title','q2_Body','q2_AcceptedAnswerBody',
                                                        'q2_AnswersBody','class']):

        self.df = pd.read_csv(path,usecols=cols_to_include)
        self.split = train_split_ratio
        with open('Cleaners.pickle') as f:
            loaded_obj = pickle.load(f)        

        punctuation_remove = loaded_obj['punctuations']
        misspelt_words = loaded_obj['mispell_dict']

    def removing_stop_words(self,text,stop_words=[]):




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




class Dataloader(Dataset):

    def __getitem__(self,path,preprocessing,target_col,rest_col):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
