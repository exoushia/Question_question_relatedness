import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.metrics import classification_report
import argparse

import tqdm
import time
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Config(object):
	embed_size = 300
	hidden_layers = 1
	hidden_size = 128
	bidirectional = True
	output_size = 4
	epochs = 25
	lr = 0.001
	ll_hidden_size = 50  # Linear layer hidden sizes
	batch_size = 32
	# max_sen_len = 20 # Sequence length for RNN
	dropout_keep = 0.2
	sample = 1
	split_ratio = 0.4
	loss_fn = nn.CrossEntropyLoss()
	patience = 25
	delta = 0.001
	batch_size_test = 32
  
 #UTILITIES:
class plot_results:
	# Instantiation of class
	def __init__(self, train_losses_plot, val_losses_plot, val_accuracies_plot,
				 figname=["Training.png", "Validation.png"], smooth=False):
		self.train_losses_plot = train_losses_plot
		self.val_losses_plot = val_losses_plot
		self.val_accuracies_plot = val_accuracies_plot
		self.smooth = smooth
		self.figname = figname

		flatten = lambda l: [item for sublist in l for item in sublist]
		if smooth:
			# If smooth is True, taking the average of every epoch
			self.train_losses_plot = np.array([np.mean(loss_iter) for loss_iter in list(self.train_losses_plot)])
			self.val_losses_plot = np.array([np.mean(loss_iter) for loss_iter in list(self.val_losses_plot)])
			self.val_accuracies_plot = np.array([np.mean(loss_iter) for loss_iter in list(self.val_accuracies_plot)])
		else:
			# If smooth is False, flattening out the list to display avg values for every 10 iterations
			self.train_losses_plot = flatten(self.train_losses_plot)
			self.val_losses_plot = flatten(self.val_losses_plot)
			self.val_accuracies_plot = flatten(self.val_accuracies_plot)

	# After instantiating the class, this is the function that needs to be called:
	def run(self, figure_sep=True):
		#        %matplotlib inline
		plt.style.use('classic')
		fig = plt.figure(figsize=(10, 8))
		print("Starting to plot figures.... \n\n")
		if figure_sep:
			ax = plt.subplot(2, 1, 1)
			ax.plot(self.train_losses_plot)
			if self.smooth:
				title_str = 'Average training loss for every Epoch'
			else:
				title_str = "Training losses for every 10 iterations"
			ax.set(ylabel='Training set values', title=title_str)
			ax.grid()
			if self.figname is not None:
				fig.savefig(self.figname[0])
			plt.show()

			ax = plt.subplot(2, 1, 2)
			ax.plot(self.val_losses_plot, color='blue')
			ax.plot(self.val_accuracies_plot, color='green')
			ax.legend(['Val losses', 'Val Accuracies'], loc='best')
			if self.smooth:
				title_str = 'Average Validation losses and accuracy for every Epoch'
			else:
				title_str = "Validation losses and accuracy for every 10 iterations"
			ax.set(ylabel='Validation set values', title=title_str)
			ax.grid()
			if self.figname is not None:
				fig.savefig(self.figname[1])
			plt.show()

		else:
			ax = plt.subplot(2, 1, 2)
			ax.plot(self.train_losses_plot, color='red')
			ax.plot(self.val_losses_plot, color='blue')
			ax.plot(self.val_accuracies_plot, color='green')
			ax.legend(['Training_loss', 'Val loss', 'Val Accuracies'], loc='best')
			if self.smooth:
				title_str = 'Average Training loss Validation losses and accuracy for every Epoch'
			else:
				title_str = "Training loss Validation loss and accuracy for every 10 iterations"
			ax.set(title=title_str)
			ax.grid()
			if self.figname is not None:
				fig.savefig(self.figname[0])
			plt.show()

		print("All graphs plotted! \n\n")


def print_classification_report(pred_list, title, target_names=['Direct', 'Duplicate', 'Indirect', 'Isolated'],
								save_result_path="Expt_results/results.csv"):

	flatten = lambda l: np.array([item for sublist in l for item in sublist])
#	pred_list = flatten(pred_list)
	print(pred_list)
	y_pred = np.array([x[0].tolist() for x in pred_list])
	y_true = np.array([x[1].tolist() for x in pred_list])
	y_pred = flatten(y_pred)
	y_true = flatten(y_true)
	print(y_true)
	str_title = "Printing Classification Report : " + title + " \n\n"
	print(str_title)
	report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
	df = pd.DataFrame(report)
	df.to_csv(save_result_path)
	print(report)

	str_title = "\n\n Printing Multilabel Confusion Matrix : " + title + " \n\n"
	print(str_title)
	print(multilabel_confusion_matrix(y_true, y_pred))

	str_title = "Printing Micro averaged scores : " + title + " \n\n"
	print(str_title)
	print(precision_recall_fscore_support(y_true, y_pred, average='micro'))
	print("\n")

	print("\n All Results Printed !! \n")


def nearest_word(OOV_word, word_embedding, word2matrix):
	try:
		OOV_word_embedding = word_embedding[OOV_word]
	except KeyError:
		OOV_word_embedding = np.random.normal(scale=0.6, size=(1, 300))
	sim_max = 0
	for i in range(len(word2matrix)):
		embedding = word2matrix[i]
		sim = cosine_similarity(embedding, OOV_word_embedding)
		if sim > sim_max:
			sim_max = sim
			index_of_closest_word = i

	return index_of_closest_word


def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
#MODEL ARCHITECTURE

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""

	def __init__(self, patience=5, verbose=False, delta=0, path_to_cpt='checkpoint.pt'):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement.
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
		"""

		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.path = path_to_cpt

	def __call__(self, val_loss, model):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score < self.best_score + self.delta:
			self.counter += 1
			# print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		'''Saves model when validation loss decrease.'''
		# if self.verbose:
		# 	print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.path)
		self.val_loss_min = val_loss


class BiLSTM(nn.Module):

	def __init__(self, config, vocab_size, word_embeddings):
		super(BiLSTM, self).__init__()

		self.config = config
		self.loss = config.loss_fn
		# self.optimizer = optimizer
		self.dropout = self.config.dropout_keep

		# Layer 1: Word2Vec Embedding.
		self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
		self.embeddings.weight = nn.Parameter(torch.as_tensor(word_embeddings, dtype=torch.float32), requires_grad=False)

		# Layer 2: Bidirectional LSTM
		self.lstm = nn.LSTM(input_size=self.config.embed_size,
							hidden_size=self.config.hidden_size,
							num_layers=self.config.hidden_layers,
							dropout=self.config.dropout_keep,
							bidirectional=True,
							batch_first=True)

		#Layer 3: Attention
		# We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
		self.W_s1 = nn.Linear(2*config.hidden_size, 350)
		self.W_s2 = nn.Linear(350, 30)
		
		# Layer 4: Rest of the layers
		self.net = nn.Sequential(nn.Linear(3, self.config.ll_hidden_size), nn.ReLU(), nn.Dropout(p=self.dropout),
								 nn.Linear(self.config.ll_hidden_size, self.config.output_size), nn.Softmax())

	# net.apply(init_weights)

	#   def init_weights(self,m):
	#     if type(m) == nn.Linear:
	#         torch.nn.init.xavier_uniform(m.weight)
	#         m.bias.data.fill_(0.01)

	# https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/selfAttention.py
	def self_attention_net(self, lstm_output):
		"""
		Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
		encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of 
		the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully 
		connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e., 
		pos & neg.
		Arguments
		---------
		lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
		---------
		Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
				  attention to different parts of the input sentence.
		Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
					  attn_weight_matrix.size() = (batch_size, 30, num_seq)
		"""
		attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix

	# https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
	def attention_net(self, lstm_output, final_state):
		""" 
		Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
		between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
		
		Arguments
		---------
		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
		final_state : Final time-step hidden state (h_n) of the LSTM
		---------	
		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.
				  
		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, seq_len)
					soft_attn_weights.size() = (batch_size, seq_len)
					new_hidden_state.size() = (batch_size, hidden_size)
		"""
		
		hidden = final_state
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state

	# https://discuss.pytorch.org/t/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers/15398/2
	# https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
	def forward(self, pairs, batch_size):
		q1 = torch.stack([x[0] for x in pairs]).to(device)
		q2 = torch.stack([x[1] for x in pairs]).to(device)

		# Input: batch_size x seq_length
		# Output: batch-size x seq_length x embedding_dimension
		x1 = self.embeddings(q1)
		x2 = self.embeddings(q2)

		# Input: (batch_size, seq_length, input_size) (input_size=embedding_dimension in this case)
		# Output: (batch_size, seq_length, 2*hidden_size) (batch_size is dim0 since batch_first is true)
		# last_hidden_state: (2 * batch_size, hidden_size)  (2 due to bidirectional, otherwise would be 1)
		# last_cell_state: (2 * batch_size, hidden_size)    (2 due to bidirectional, otherwise would be 1)
		lstm_out1, (h_n1, c_n1) = self.lstm(x1.to(device))
		lstm_out2, (h_n2, c_n2) = self.lstm(x2.to(device))

		#Self Attention
		attn_weight_matrix1 = self.self_attention_net(lstm_out1)
		attn_weight_matrix2 = self.self_attention_net(lstm_out2)
		# attn_weight_matrix: (batch_size, r, seq_len)
		hidden_matrix1 = torch.bmm(attn_weight_matrix1, lstm_out1)
		hidden_matrix2 = torch.bmm(attn_weight_matrix2, lstm_out2)
		# hidden_matrix: (batch_size, r, 2*hidden_size)

		# print("Shape of hidden state is {} before concat".format(h_n1.shape))

		# Concating both iterations of bilstm
		h_n1 = torch.cat([h_n1[0, :, :], h_n1[1, :, :]], -1).view(batch_size, 2 * self.config.hidden_size)
		h_n2 = torch.cat([h_n2[0, :, :], h_n2[1, :, :]], -1).view(batch_size, 2 * self.config.hidden_size)

		# Attention
		attn_h_n1 = self.attention_net(lstm_out1, h_n1)
		attn_h_n2 = self.attention_net(lstm_out1, h_n2)

		# print("Shape of hidden state is {} after concat and reshape".format(h_n1.shape))
		
		# shape of hidden state = batch_size,2*hidden_size -> dot product across second dimension
		dotproduct = torch.sum(torch.mul(attn_h_n1, attn_h_n2), 1).view(batch_size, -1)
		# Shape of h_n1 => batch_size,2*hidden_size

		return dotproduct

	def calling(self, t, b, a, batch_size):
		inner_dot_titles = self.forward(t, batch_size)
		inner_dot_body = self.forward(b, batch_size)
		inner_dot_ans = self.forward(a, batch_size)

		# need to concatenate these tensors along the right dimention - batch size
		concat_input_to_dense = torch.cat((inner_dot_titles, inner_dot_body, inner_dot_ans), 1)

		# concat_input_to_dense = concat_input_to_dense.view(-1,)
		output = self.net(concat_input_to_dense)
		return output.view(-1, self.config.output_size)


#PREPROCESS

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

	def encoding_target(self, df):
		print("-> Encoding target.../n")
		label_enc = LabelEncoder()
		df[self.target_col] = label_enc.fit_transform(df[self.target_col])
		return df

	def replace_encoding(self, num):
		if num == 0: return np.array([1, 0, 0, 0]).astype('int64')
		if num == 1: return np.array([0, 1, 0, 0]).astype('int64')
		if num == 2: return np.array([0, 0, 1, 0]).astype('int64')
		if num == 3: return np.array([0, 0, 0, 1]).astype('int64')

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

		self.df = self.encoding_target(self.df)
		self.df[self.target_col] = self.df[self.target_col].apply(lambda x: self.replace_encoding(int(x)))

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
				for item in self.df[column]:
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
				for sent in self.df[col].iloc[i].split('.'):
					result = result + self.extracting_indices(self.vocab, sent.lower())

				result_word2index.append(result)
			self.df[col] = result_word2index
		print("Trimming columns...")
		self.df = self.trim(self.mapping_trimsize, self.df)

		for col in self.df.columns:
			self.df[col] = self.df[col].apply(lambda x: np.array(x).astype('int64').squeeze())

		print("dtype ", type(self.df['q1_Title'].iloc[2]))

		return self.df, self.vocab



#TRAIN
def embeddings_gen(vocab, path_to_glove):
	matrix_len = len(vocab.word2index)
	weights_matrix = np.zeros((matrix_len + 1, 300))
	words_found = 0

	#     infile = open(path_to_glove,'rb')
	#     glove = pickle.load(infile)
	#     infile.close()
	glove = KeyedVectors.load_word2vec_format(path_to_glove, binary=False)

	for index in vocab.index2word:
		if index == 0:
			weights_matrix[index] = np.zeros((1, 300))
		try:
			weights_matrix[index] = glove[vocab.index2word[index]]
			words_found = words_found + 1
		except KeyError:
			weights_matrix[index] = np.random.normal(scale=0.6, size=(1, 300))

	embedding_matrix = torch.FloatTensor(weights_matrix)
	print("Words found in embedding:" + str(words_found))

	del glove

	return embedding_matrix


def data_loading(train_path, val_path, preprocess, target, config,
				 rest_col=['id', 'q1_Title', 'q1_Body', 'q1_AcceptedAnswerBody',
						   'q1_AnswersBody', 'q2_Title', 'q2_Body', 'q2_AcceptedAnswerBody',
						   'q2_AnswersBody'],
				 mapping_trimsize={'q1_Title': 10, 'q1_Body': 60, 'answer_text1': 180, 'q2_Title': 10, 'q2_Body': 60,
								   'answer_text2': 180}):
	if val_path is None:
		if preprocess:
			print(rest_col.append(target))
			preprocess_class = Preprocessing(train_path, target)
			df, new_cols = preprocess_class.run()
			print(" Writing preprocessed data for future use..")
			df.to_csv(train_path[:-4] + "_preprocessed.csv")
		else:
			rest_col=["id", "q1_Title", "q2_Title", "q1_Body", "q2_Body", "answer_text1", "answer_text2", "class"]
			df = pd.read_csv(train_path, usecols=rest_col)

		vocab = Vocab('stack')

		batchify_obj = forming_batches(vocab, mapping_trimsize, df, target)

		df, vocab = batchify_obj.run()
		print(df.head())

		print("\n\n Sequence of columns : ")
		rest_col = [col for col in list(df.columns) if col not in ['id']]
		print(rest_col[0:2].append(rest_col[-1]))

		dataset_title = Bilstm_Dataset(df, rest_col[0:2], rest_col[-1])
		dataset_body = Bilstm_Dataset(df, rest_col[2:4], rest_col[-1])
		dataset_answer = Bilstm_Dataset(df, rest_col[4:6], rest_col[-1])

		NUM_INSTANCES = dataset_title.__len__()
		NUM_INSTANCES = NUM_INSTANCES * config.sample
		TEST_SIZE = int(NUM_INSTANCES * config.split_ratio)

		num_batches_train = (NUM_INSTANCES - TEST_SIZE) / config.batch_size
		num_batches_val = TEST_SIZE / config.batch_size

		indices = list(range(NUM_INSTANCES))

		val_idx = np.random.choice(indices, size=TEST_SIZE, replace=False)
		train_idx = list(set(indices) - set(val_idx))
		train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

		train_loader_title = DataLoader(dataset_title, batch_size=config.batch_size, sampler=train_sampler)
		val_loader_title = DataLoader(dataset_title, batch_size=config.batch_size, sampler=val_sampler)

		train_loader_body = DataLoader(dataset_body, batch_size=config.batch_size, sampler=train_sampler)
		val_loader_body = DataLoader(dataset_body, batch_size=config.batch_size, sampler=val_sampler)

		train_loader_ans = DataLoader(dataset_answer, batch_size=config.batch_size, sampler=train_sampler)
		val_loader_ans = DataLoader(dataset_answer, batch_size=config.batch_size, sampler=val_sampler)

		train_loaders = [train_loader_title, train_loader_body, train_loader_ans]
		val_loaders = [val_loader_title, val_loader_body, val_loader_ans]

		del train_loader_title, train_loader_body, train_loader_ans, val_loader_title, val_loader_body, val_loader_ans
		del dataset_title, dataset_body, dataset_answer

		return train_loaders, val_loaders, vocab, int(num_batches_train), int(num_batches_val)
	else:
		if preprocess:
			print(rest_col.append(target))
			preprocess_class = Preprocessing(train_path, target)
			df, new_cols = preprocess_class.run()

			preprocess_class_val = Preprocessing(val_path, target)
			df_val, new_cols = preprocess_class_val.run()

			print(" Writing preprocessed data for future use..")
			df.to_csv(train_path[:-4] + "_preprocessed.csv")
			df_val.to_csv(val_path[:-4] + "_preprocessed.csv")

		else:
			rest_col=["id", "q1_Title", "q2_Title", "q1_Body", "q2_Body", "answer_text1", "answer_text2", "class"]			
			df = pd.read_csv(train_path, usecols=rest_col)
			df_val = pd.read_csv(val_path, usecols=rest_col)

		vocab = Vocab('stack')

		batchify_obj = forming_batches(vocab, mapping_trimsize, df, target, vocab_new=True)
		df, vocab = batchify_obj.run()

		batchify_obj_val = forming_batches(vocab, mapping_trimsize, df_val, target, vocab_new=True)
		df_val, vocab = batchify_obj_val.run()

		print(df.head())

		print("\n\n Sequence of columns : ")
		rest_col = [col for col in list(df.columns) if col not in ['id']]
		print(rest_col)

		dataset_title = Bilstm_Dataset(df, rest_col[0:2], rest_col[-1])
		dataset_body = Bilstm_Dataset(df, rest_col[2:4], rest_col[-1])
		dataset_answer = Bilstm_Dataset(df, rest_col[4:6], rest_col[-1])

		dataset_title_val = Bilstm_Dataset(df_val, rest_col[0:2], rest_col[-1])
		dataset_body_val = Bilstm_Dataset(df_val, rest_col[2:4], rest_col[-1])
		dataset_answer_val = Bilstm_Dataset(df_val, rest_col[4:6], rest_col[-1])

		NUM_INSTANCES_TRAIN = dataset_title.__len__()
		NUM_INSTANCES_TRAIN = NUM_INSTANCES_TRAIN * config.sample
		NUM_INSTANCES_VAL = dataset_title_val.__len__()
		NUM_INSTANCES_VAL = NUM_INSTANCES_VAL * config.sample

		num_batches_train = NUM_INSTANCES_TRAIN / config.batch_size
		num_batches_val = NUM_INSTANCES_VAL / config.batch_size

		indices_train = list(range(NUM_INSTANCES_TRAIN))
		indices_val = list(range(NUM_INSTANCES_VAL))

		val_idx = np.random.choice(indices_val, size=NUM_INSTANCES_VAL, replace=False)
		train_idx = np.random.choice(indices_train, size=NUM_INSTANCES_TRAIN, replace=False)
		train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

		train_loader_title = DataLoader(dataset_title, batch_size=config.batch_size, sampler=train_sampler)
		val_loader_title = DataLoader(dataset_title_val, batch_size=config.batch_size, sampler=val_sampler)

		train_loader_body = DataLoader(dataset_body, batch_size=config.batch_size, sampler=train_sampler)
		val_loader_body = DataLoader(dataset_body_val, batch_size=config.batch_size, sampler=val_sampler)

		train_loader_ans = DataLoader(dataset_answer, batch_size=config.batch_size, sampler=train_sampler)
		val_loader_ans = DataLoader(dataset_answer_val, batch_size=config.batch_size, sampler=val_sampler)

		train_loaders = [train_loader_title, train_loader_body, train_loader_ans]
		val_loaders = [val_loader_title, val_loader_body, val_loader_ans]

		del train_loader_title, train_loader_body, train_loader_ans, val_loader_title, val_loader_body, val_loader_ans
		del dataset_title, dataset_body, dataset_answer, dataset_title_val, dataset_body_val, dataset_answer_val

		return train_loaders, val_loaders, vocab, int(num_batches_train), int(num_batches_val)


def evaluate_model(model, loader, num_batches, batch_size):
	title_iter = iter(loader[0])
	body_iter = iter(loader[1])
	ans_iter = iter(loader[2])

	running_loss = 0
	running_corrects = 0
	pred_true = []
	for i in range(num_batches):
		title = next(title_iter)  # ith batch
		body = next(body_iter)  # ith batch
		ans = next(ans_iter)  # ith batch

		y_pred = model.calling(title[0].to(device), body[0].to(device), ans[0].to(device), batch_size)
		labels = title[1]

		# print("Shape of y pred2 {}".format(y_pred.shape))
		# print("Shape of y true2 {}".format(labels.shape))

		pred_true.append([torch.argmax(y_pred, 1), torch.argmax(labels, 1)])

		loss = model.loss(y_pred, torch.argmax(labels, 1))
		running_loss = running_loss + loss.item()
		running_corrects = running_corrects + torch.sum(torch.argmax(y_pred, 1) == torch.argmax(labels, 1)).item()

	epoch_loss = 1.0 * running_loss / num_batches
	epoch_acc = 1.0 * running_corrects / num_batches

	return epoch_loss, epoch_acc, pred_true


def run_train(model, train_loader, val_loader, epoch, num_batches_train, num_batches_val, batch_size, optimizer):
	train_losses = []
	val_accuracies = []
	val_losses = []
	f1 = []
	losses = []

	title_iter = iter(train_loader[0])
	body_iter = iter(train_loader[1])
	ans_iter = iter(train_loader[2])

	model.train()
	for i in tqdm.trange(num_batches_train, file=sys.stdout, desc='Iterations'):
		optimizer.zero_grad()

		title = next(title_iter)  # ith batch
		body = next(body_iter)  # ith batch
		ans = next(ans_iter)  # ith batch

		# print("Shape of train title iter {}".format(title[0].shape))

		y_pred = model.calling(title[0].to(device), body[0].to(device), ans[0].to(device), batch_size)
		y_true = title[1]

		# print("Shape of y pred {}".format(y_pred.shape))
		# print("Shape of y true {}".format(y_true.shape))

		loss = model.loss(y_pred, torch.argmax(y_true, 1))
		loss.backward()
		losses.append(loss.detach().cpu().numpy())
		optimizer.step()

		if i % 300 == 0:
			print("Iter: {},Epoch: {}\n".format(i + 1, epoch))
			avg_train_loss = np.mean(losses)
			train_losses.append(avg_train_loss)
			print("\tAverage training loss: {:.5f}\n".format(avg_train_loss))
			losses = []
			model.eval()
			# Evalute Accuracy on validation set
			with torch.no_grad():
				val_loss, val_accuracy, curr_F1 = evaluate_model(model, val_loader, num_batches_val, batch_size)
				val_accuracies.append(val_accuracy)
				val_losses.append(val_loss)
				f1.append(curr_F1)

				print("\tVal Accuracy: {:.4f}\n".format(val_accuracy))
				print("\n")
				model.train()

	return train_losses, val_losses, val_accuracies, f1


def train_model(path_to_data, path_vocab_save, path_embed_matrix_save, train_file, val_file, path_to_glove,
				path_to_cpt, config, preprocess=False):
	train_path = path_to_data + '/' + train_file
	if val_file is None:
		val_path = None
	else:
		val_path = path_to_data + '/' + val_file

	train_loaders, val_loaders, vocab, num_batches_train, num_batches_val = data_loading(train_path, val_path,
																						 preprocess, target='class',
																						 config=config)

	best_val_loss = float("inf")

	embedding_matrix = embeddings_gen(vocab, path_to_glove)

	# Saving vocab object and embedding matrix
	save_object(vocab, path_vocab_save)
	save_object(embedding_matrix, path_embed_matrix_save)

	model = BiLSTM(config, len(vocab.word2index), embedding_matrix)
	optimizer = optim.Adam(model.parameters(), lr=config.lr)

	# initialize the early_stopping object
	early_stopping = EarlyStopping(patience=config.patience, verbose=True, delta=config.delta, path_to_cpt=path_to_cpt)

	if torch.cuda.is_available():
		model.to(device)

	model.train()

	train_losses_plot = []
	val_accuracies_plot = []
	val_losses_plot = []
	epoch_f1 = []

	start_of_training = time.time()

	for i in range(config.epochs):
		print("Epoch: {}".format(i))
		train_loss, val_loss, val_accuracy, all_F1 = run_train(model, train_loaders, val_loaders, i, num_batches_train,
															   num_batches_val, config.batch_size, optimizer)

		train_losses_plot.append(train_loss)
		val_losses_plot.append(val_loss)
		val_accuracies_plot.append(val_accuracy)
		epoch_f1.append(all_F1)

		early_stopping(val_loss[-1], model)

		if early_stopping.early_stop:
			print("Early stopping....\n")
			break

		avg_train_losses = np.mean(np.array(train_loss))
		avg_val_losses = np.mean(np.array(val_loss))
		if avg_val_losses < best_val_loss:
			best_val_loss = avg_val_losses
			best_model = model

	end_of_training = time.time()

	print("\n\n Training Time : {:5.2f} secs".format(end_of_training - start_of_training))

	# load the last checkpoint with the best model
	model.load_state_dict(torch.load(path_to_cpt))

	return model, avg_train_losses, avg_val_losses, train_losses_plot, val_accuracies_plot, val_losses_plot, epoch_f1, vocab, embedding_matrix



#Arguments for path and other boolean values: <Can be changed for running on colab/kaggle>
path_to_data     = Data
path_vocab_save   = Expt_results/Vocab.pkl
path_embed_matrix        = Expt_results/EmbedMatrix.pkl
path_to_cpt        = Expt_results/checkpoints/checkpoint.pt
path_to_glove       = Data/glove.840B.300d.word2vec.txt
model_path = 
save_result_path = Expt_results/results.csv
name_train = train_sample.csv
name_val = None
name_test = train_sample.csv
target_names = ['Direct', 'Duplicate', 'Indirect', 'Isolated']
figname = ['Training.png', 'Validation.png']
mode = train_&_test
to_preprocess = True
smooth = False

#setting seeds for reproducibility
torch.manual_seed(100)
if torch.cuda.is_available():
  torch.cuda.manual_seed(100)
np.random.seed(100)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
#	device="cpu"
config = Config()

test_path = path_to_data + '/' + name_test

if mode == "train_&_test":
    model, avg_train_losses, avg_val_losses, train_losses_plot, val_accuracies_plot, val_losses_plot, epoch_f1, vocab, embedding_matrix = train_model(
    path_to_data, path_vocab_save, path_embed_matrix, name_train, name_val,
    path_to_glove, path_to_cpt, config, to_preprocess)
    plot = plot_results(train_losses_plot, val_losses_plot, val_accuracies_plot, figname, smooth)
#		plot.run(figure_sep=True)
elif mode == "only_test":
  # unpickling vocab and embed_metrix
  infile = open(path_vocab_save, 'rb')
  vocab = pickle.load(infile)
  infile.close()

  infile = open(args.path_embed_matrix, 'rb')
  embedding_matrix = pickle.load(infile)
  infile.close()

  # Unloading the best model saved in last session
  model = BiLSTM(config, len(vocab.word2index), embedding_matrix).to(device)
  model.load_state_dict(torch.load(args.model_path))
  model.eval()

torch.cuda.empty_cache()
test_loss, test_acc, test_pred_true = run_test(test_path, model, vocab, embedding_matrix, config,
                         args.to_preprocess, target='class')

# Only for Test rn - we can modify later
print_classification_report(test_pred_true, args.title, args.target_names, args.save_result_path)


