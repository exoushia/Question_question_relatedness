from sklearn.metrics import classification_report
import argparse, time
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

# CUDA for PyTorch
device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")

from Model.network_architecture import *
from train import train_model
from utils import *
from test import *

torch.backends.cudnn.benchmark = True
timestamp = time.strftime("%d-%m-%Y_%H:%M")
print(timestamp)

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



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-path_to_data", default='Data', type=str, help="path to data folder")
	parser.add_argument("-path_vocab_save", default='Expt_results/Vocab.pkl', type=str,
						help="path to pretrained vocab object")
	parser.add_argument("-path_embed_matrix", default='Expt_results/EmbedMatrix.pkl', type=str,
						help="path to preformed word_embedding matrix")
	parser.add_argument("-path_to_model", default='Expt_results/checkpoints/checkpoint=' + timestamp + '.pt',
						help="Path to where checkpoints will be stored")
	parser.add_argument("-path_to_glove", default='Data/glove.840B.300d.word2vec.txt', help="Path to word embeddings")
	# parser.add_argument("-model_path", default='Expt_results/checkpoints/checkpoint.pt', help="Path to the trained model for mode:only test")
	parser.add_argument("-save_result_path", default="Expt_results/results=" + timestamp + ".csv",
						help="Path to save results on test")
	parser.add_argument("-name_train", default='train_sample.csv', help="Name of train file")
	parser.add_argument("-name_val", default='val_sample.csv', help="Name of val file or None: Train will be splitted")
	parser.add_argument("-name_test", default='test_sample.csv', type=str, help="Name of test file")
	parser.add_argument("-to_preprocess_train", default=False, type=bool, help="Boolean to preprocess train and val data")
	parser.add_argument("-to_preprocess_test", default=False, type=bool, help="Boolean to preprocess test data")
	parser.add_argument("-target_names", default=['Direct', 'Duplicate', 'Indirect', 'Isolated'], type=list,
						help="classes")
	parser.add_argument("-figname", default=["Training=" + timestamp + ".png", "Validation=" + timestamp + ".png"], type=list,
						help="Names of images to be stored, None : if need not be saved")
	parser.add_argument("-title", default="Test Set", type=str, help="Title of the Results' Report")
	parser.add_argument("-mode", default='train_&_test', type=str, choices=['train_&_test', 'only_test'])
	parser.add_argument("-smooth", default=False, type=bool, help="")

	args = parser.parse_args()
	
	#setting seeds for reproducibility
	torch.manual_seed(100)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(100)
	np.random.seed(100)

	config = Config()

	# 	path_to_data='Data'
	# 	train_file='train_sample.csv'
	# 	val_file='val_sample.csv'
	# 	test_file='test_sample.csv'
	# 	path_to_glove='Data/glove.840B.300d.word2vec.txt'
	# 	path_to_cpt='Expt_results/checkpoints/checkpoint.pt'
	# 	save_result_path="Expt_results/results.csv"
	# 	to_preprocess = True
	# 	smooth = False
	# 	figname=["Training.png","Validation.png"]
	# 	target_names=['Direct', 'Duplicate', 'Indirect','Isolated']
	# 	title = "Test Set"

	test_path = args.path_to_data + '/' + args.name_test

	if args.mode == "train_&_test":
		model, avg_train_losses, avg_val_losses, train_losses_plot, val_accuracies_plot, val_losses_plot, epoch_f1, vocab, embedding_matrix = train_model(
			args.path_to_data, args.path_vocab_save, args.path_embed_matrix, args.name_train, args.name_val,
			args.path_to_glove, args.path_to_model, config, args.to_preprocess_train)
		plot = plot_results(train_losses_plot, val_losses_plot, val_accuracies_plot, args.figname, args.smooth)
		plot.run(figure_sep=True)
	elif args.mode == "only_test":
		# unpickling vocab and embed_metrix
		infile = open(args.path_vocab_save, 'rb')
		vocab = pickle.load(infile)
		infile.close()

		infile = open(args.path_embed_matrix, 'rb')
		embedding_matrix = pickle.load(infile)
		infile.close()

		# Unloading the best model saved in last session
		model = BiLSTM(config, len(vocab.word2index), embedding_matrix).to(device)
		model.load_state_dict(torch.load(args.path_to_model))
		model.eval()

	torch.cuda.empty_cache()
	test_loss, test_acc, test_pred_true = run_test(test_path, model, vocab, embedding_matrix, config,
												   args.to_preprocess_test, target='class')

	# Only for Test rn - we can modify later
	print_classification_report(test_pred_true, args.title, args.target_names, args.save_result_path)
