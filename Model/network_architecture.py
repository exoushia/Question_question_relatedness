# https://discuss.pytorch.org/t/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers/15398/2
# https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np

torch.backends.cudnn.benchmark = True

sys.path.append(os.path.realpath('..'))
from settings import device


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
		self.embeddings.weight = nn.Parameter(torch.as_tensor(word_embeddings, dtype=torch.float32),
											  requires_grad=False)

		# Layer 2: Bidirectional LSTM
		self.lstm = nn.LSTM(input_size=self.config.embed_size,
							hidden_size=self.config.hidden_size,
							num_layers=self.config.hidden_layers,
							dropout=self.config.dropout_keep,
							bidirectional=True,
							batch_first=True)

		# Layer 3: Attention
		# We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
		self.W_s1 = nn.Linear(2 * config.hidden_size, 350)
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
    def similarity(self, pairs, batch_size):
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

		# Self Attention
		# attn_weight_matrix1 = self.self_attention_net(lstm_out1)
		# attn_weight_matrix2 = self.self_attention_net(lstm_out2)
		# # attn_weight_matrix: (batch_size, r, seq_len)
		# hidden_matrix1 = torch.bmm(attn_weight_matrix1, lstm_out1)
		# hidden_matrix2 = torch.bmm(attn_weight_matrix2, lstm_out2)
		# hidden_matrix: (batch_size, r, 2*hidden_size)

		# print("Shape of hidden state is {} before concat".format(h_n1.shape))

		# Concating both iterations of bilstm
		h_n1 = torch.cat([h_n1[0, :, :], h_n1[1, :, :]], -1).view(batch_size, 2 * self.config.hidden_size)
		h_n2 = torch.cat([h_n2[0, :, :], h_n2[1, :, :]], -1).view(batch_size, 2 * self.config.hidden_size)

		# Attention
		# h_n1 = self.attention_net(lstm_out1, h_n1)
		# h_n2 = self.attention_net(lstm_out1, h_n2)

		# print("Shape of hidden state is {} after concat and reshape".format(h_n1.shape))

		# shape of hidden state = batch_size,2*hidden_size -> dot product across second dimension
		dotproduct = torch.sum(torch.mul(h_n1, h_n2), 1).view(batch_size, -1)
		# Shape of h_n1 => batch_size,2*hidden_size

		return dotproduct

    def forward(self, t, b, a, batch_size):
        inner_dot_titles = self.similarity(t, batch_size)
        inner_dot_body = self.similarity(b, batch_size)
        inner_dot_ans = self.similarity(a, batch_size)

		# need to concatenate these tensors along the right dimention - batch size
		concat_input_to_dense = torch.cat((inner_dot_titles, inner_dot_body, inner_dot_ans), 1)

		# concat_input_to_dense = concat_input_to_dense.view(-1,)
		output = self.net(concat_input_to_dense)
		return output.view(-1, self.config.output_size)



# https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
class CNN_classifier(nn.Module):
	"""An 1D Convulational Neural Network for Sentence Classification."""
	def __init__(self, pretrained_embedding=None, freeze_embedding=False, vocab_size=None, embed_dim=300, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], num_classes=4, dropout=0.5):
		"""
		The constructor for CNN_classifier class.
		Args:
			pretrained_embedding (torch.Tensor): Pretrained embeddings with
				shape (vocab_size, embed_dim)
			freeze_embedding (bool): Set to False to fine-tune pretraiend
				vectors. Default: False
			vocab_size (int): Need to be specified when not pretrained word
				embeddings are not used.
			embed_dim (int): Dimension of word vectors. Need to be specified
				when pretrained word embeddings are not used. Default: 300
			filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
			num_filters (List[int]): List of number of filters, has the same
				length as `filter_sizes`. Default: [100, 100, 100]
			n_classes (int): Number of classes. Default: 2
			dropout (float): Dropout rate. Default: 0.5
		"""

		super(CNN_classifier, self).__init__()
		# Embedding layer
		if pretrained_embedding is not None:
			self.vocab_size, self.embed_dim = pretrained_embedding.shape
			self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
		else:
			self.embed_dim = embed_dim
			self.embeddings = nn.Embedding(num_embeddings=vocab_size,
										  embedding_dim=self.embed_dim,
										  padding_idx=0,
										  max_norm=5.0)

		# Layer 1: Word2Vec Embedding.
		# self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
			self.embeddings.weight = nn.Parameter(torch.as_tensor(word_embeddings, dtype=torch.float32),
											  requires_grad=False)

		# Conv Network
		self.conv1d_list = nn.ModuleList(
			[nn.Conv1d(in_channels=self.embed_dim, out_channels=num_filters[i], kernel_size=filter_sizes[i]) for i in range(len(filter_sizes))]
			)
		# Fully-connected layer and Dropout
		self.fc = nn.Linear(np.sum(num_filters), num_classes)
		self.dropout = nn.Dropout(p=dropout)


	def forward(self, input_ids):
		"""Perform a forward pass through the network.

		Args:
			input_ids (torch.Tensor): A tensor of token ids with shape
				(batch_size, max_sent_length)

		Returns:
			logits (torch.Tensor): Output logits with shape (batch_size,
				n_classes)
		"""

		# Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
		x_embed = self.embedding(input_ids).float()

		# Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
		# Output shape: (b, embed_dim, max_len)
		x_reshaped = x_embed.permute(0, 2, 1)

		# Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
		x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

		# Max pooling. Output shape: (b, num_filters[i], 1)
		x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
			for x_conv in x_conv_list]
		
		# Concatenate x_pool_list to feed the fully connected layer.
		# Output shape: (b, sum(num_filters))
		x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
						 dim=1)
		
		# Compute logits. Output shape: (b, n_classes)
		logits = self.fc(self.dropout(x_fc))

		return logits



class CNNSentence(nn.Module):

	def __init__(self, args, data, vectors):
		super(CNNSentence, self).__init__()

		self.args = args

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim, padding_idx=1)
		# initialize word embedding with pretrained word2vec
		if args.mode != 'rand':
			self.word_emb.weight.data.copy_(torch.from_numpy(vectors))
		if args.mode in ('static', 'multichannel'):
			self.word_emb.weight.requires_grad = False
		if args.mode == 'multichannel':
			self.word_emb_multi = nn.Embedding(args.word_vocab_size, args.word_dim, padding_idx=1)
			self.word_emb_multi.weight.data.copy_(torch.from_numpy(vectors))
			self.in_channels = 2
		else:
			self.in_channels = 1

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		for filter_size in args.FILTER_SIZES:
			conv = nn.Conv1d(self.in_channels, args.num_feature_maps, args.word_dim * filter_size, stride=args.word_dim)
			setattr(self, 'conv_' + str(filter_size), conv)

		self.fc = nn.Linear(len(args.FILTER_SIZES) * 100, args.class_size)

	def forward(self, batch):
		x = batch.text
		batch_size, seq_len = x.size()

		conv_in = self.word_emb(x).view(batch_size, 1, -1)
		if self.args.mode == 'multichannel':
			conv_in_multi = self.word_emb_multi(x).view(batch_size, 1, -1)
			conv_in = torch.cat((conv_in, conv_in_multi), 1)

		conv_result = [
			F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(conv_in)), seq_len - filter_size + 1).view(-1,
																													self.args.num_feature_maps)
			for filter_size in self.args.FILTER_SIZES]

		out = torch.cat(conv_result, 1)
		out = F.dropout(out, p=self.args.dropout, training=self.training)
		out = self.fc(out)

		return out
