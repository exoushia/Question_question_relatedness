from train import train_model
from utils import *
import torch.nn as nn
from sklearn.metrics import classification_report


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
    


if __name__ == '__main__':

	config = Config()

	path_to_data=''
	train_file='train_sample.csv'
	val_file=''
	test_file=''
	path_to_glove='glove.840B.300d.pkl'
	path_to_cpt='Expt_results/checkpoints/checkpoint.pt'
	figname=["Training.png","Validation.png"]
	preprocess = False
	smooth = False
	test_path = path_to_data + '/' + test_file
	save_result_path="Expt_results/results.csv")
	target_names=['class 0', 'class 1', 'class 2','class 3'],
	title = "Test Set"

    model, avg_train_losses, avg_val_losses, train_losses_plot, val_accuracies_plot, val_losses_plot, epoch_f1, vocab = train_model(path_to_data, train_file, val_file, test_file, path_to_glove, path_to_cpt, config,preprocess)
	
	plot_results(train_losses_plot,val_losses_plot,val_accuracies_plot,figname,smooth)
	
	test_loss , test_acc , maintaining_F1 = run_test(test_path,preprocess,vocab,model,target='class')
	
	#Only for Test rn - we can modify later 
	print_classification_report(maintaining_F1,title,target_names,save_result_path)
	
	
	
	
