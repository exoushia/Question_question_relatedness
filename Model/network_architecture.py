# https://discuss.pytorch.org/t/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers/15398/2
# https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np


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
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
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
		self.embeddings.weight = nn.Parameter(torch.tensor(word_embeddings, dtype=torch.float32), requires_grad=False)

		# Layer 2: Bidirectional LSTM
		self.lstm = nn.LSTM(input_size=self.config.embed_size,
							hidden_size=self.config.hidden_size,
							num_layers=self.config.hidden_layers,
							dropout=self.config.dropout_keep,
							bidirectional=True,
							batch_first=True)

		# Layer 3: Attention Layer
		# self.attn = nn.Dense(2*self.config.hidden_size, 1)

		# Layer 4: Rest of the layers
		self.net = nn.Sequential(nn.Linear(3, self.config.ll_hidden_size), nn.ReLU(), nn.Dropout(p=self.dropout),
								 nn.Linear(self.config.ll_hidden_size, self.config.output_size), nn.Softmax())

	# net.apply(init_weights)

	#   def init_weights(self,m):
	#     if type(m) == nn.Linear:
	#         torch.nn.init.xavier_uniform(m.weight)
	#         m.bias.data.fill_(0.01)

	# https://discuss.pytorch.org/t/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers/15398/2
	# https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
	def forward(self, pairs, batch_size):
		q1 = torch.stack([x[0] for x in pairs]).cuda()
		q2 = torch.stack([x[1] for x in pairs]).cuda()

		# Input: batch_size x seq_length
		# Output: batch-size x seq_length x embedding_dimension
		x1 = self.embeddings(q1)
		x2 = self.embeddings(q2)

		# Input: seq_length * batch_size * input_size (embedding_dimension in this case)
		# Output: seq_length * batch_size * hidden_size
		# last_hidden_state: batch_size * hidden_size
		# last_cell_state: batch_size * hidden_size
		lstm_out1, (h_n1, c_n1) = self.lstm(x1.cuda())
		lstm_out2, (h_n2, c_n2) = self.lstm(x2.cuda())

		print("Shape of hidden state is {} before concat".format(h_n1.shape))

		# Concating both iterations of bilstm
		h_n1 = torch.cat([h_n1[0, :, :], h_n1[1, :, :]], -1).view(batch_size, 2 * self.config.hidden_size)
		h_n2 = torch.cat([h_n2[0, :, :], h_n2[1, :, :]], -1).view(batch_size, 2 * self.config.hidden_size)

		print("Shape of hidden state is {} after concat and reshape".format(h_n1.shape))

		# weights = []
		# for i in range(2*self.config.hidden_size):
		#   weights.append(self.attn(h_n1))
		# normalized_weights = F.softmax(torch.cat(weights, 1), 1)
		# attn_applied = torch.bmm(normalized_weights.unsqueeze(0), h_n1.view(1, 2*self.config.hidden_size, -1))
		# #If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p)

		# weights2 = []
		# for i in range(2*self.config.hidden_size):
		#   weights2.append(self.attn(h_n2))
		# normalized_weights2 = F.softmax(torch.cat(weights2, 1), 1)

		# attn_applied2 = torch.bmm(normalized_weights2.unsqueeze(0), h_n2.view(1, 2*self.config.hidden_size, -1))

		# shape of hidden state = batch_size,hidden_dimension*2 -> dot product across second dimension
		# dotproduct = torch.tensor(np.sum(np.multiply(a.numpy(),b.numpy()),axis=1,keepdims=False),dtype=torch.float32)
		dotproduct = torch.sum(torch.mul(h_n1, h_n2), 1).view(batch_size, -1)
		# Shape of h_n1 => batch_size,hidden_dim*2

		# print(dotproduct.shape)  #32x1
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
