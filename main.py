from train import train_model

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

    train_model(path_to_data, train_file, val_file, test_file, path_to_glove, path_to_cpt, config,preprocess=False)
