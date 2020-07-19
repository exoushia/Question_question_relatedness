import tqdm
import time
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim

from Model.network_architecture import *
from Model.data_loader import *
from utils import *

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


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

def encoding_target(df, target_col):
	print("-> Encoding target.../n")
	label_enc = LabelEncoder()
	df[target_col] = label_enc.fit_transform(df[target_col])
	return df

def replace_encoding(num):
	if num == 0: return np.array([1, 0, 0, 0]).astype('int64')
	if num == 1: return np.array([0, 1, 0, 0]).astype('int64')
	if num == 2: return np.array([0, 0, 1, 0]).astype('int64')
	if num == 3: return np.array([0, 0, 0, 1]).astype('int64')

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
			df.to_csv(train_path[:-4] + "_preprocessed.csv", index=False)
		else:
			rest_col=["id", "q1_Title", "q2_Title", "q1_Body", "q2_Body", "answer_text1", "answer_text2", target]
			df = pd.read_csv(train_path, usecols=rest_col)

		vocab = Vocab('stack')

		df = encoding_target(df, target)
		df[target] = df[target].apply(lambda x: replace_encoding(int(x)))

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
			df.to_csv(train_path[:-4] + "_preprocessed.csv", index=False)
			df_val.to_csv(val_path[:-4] + "_preprocessed.csv", index=False)

		else:
			rest_col=["id", "q1_Title", "q2_Title", "q1_Body", "q2_Body", "answer_text1", "answer_text2", target]			
			df = pd.read_csv(train_path, usecols=rest_col)
			df_val = pd.read_csv(val_path, usecols=rest_col)

		vocab = Vocab('stack')

		df = encoding_target(df, target)
		df[target] = df[target].apply(lambda x: replace_encoding(int(x)))

		df_val = encoding_target(df_val, target)
		df_val[target] = df_val[target].apply(lambda x: replace_encoding(int(x)))

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
