import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import sys
import numpy as np

from Model.network_architecture import BiLSTM, EarlyStopping


from Model/data_loader import *

class Config(object):
    embed_size = 300
    hidden_layers = 1
    hidden_size = 128
    bidirectional = True
    output_size = 4
    epochs = 25
    lr = 0.001
    batch_size = 32
    # max_sen_len = 20 # Sequence length for RNN
    dropout_keep = 0.2
    sample = 1.0
    patience = 5
	delta = 0.01



def evaluate_model(model, val_iterator,num_batches_val,loss_fn):
    all_y = []
    titles = val_iterator[0]
    body = val_iterator[1]
    ans = val_iterator[2]
    running_loss = 0
    running_corrects = 0
    maintaining_F1 = []
    for i in range(num_batches_val):
        y_pred = model.calling([titles[0][i],titles[1][i]],[body[0][i],body[1][i]],[ans[0][i],ans[1][i]])
        _ , predicted = torch.max(y_pred,1) 
        labels = titles[2][i]
        maintaining_F1.append([predicted,labels])
        # _,label_idx = torch.max(titles[2][i])
        loss = loss_fn(y_pred,torch.tensor(labels , dtype=torch.long)) 
        running_loss =running_loss+ loss.item()
        running_corrects = running_corrects+ torch.sum(predicted == labels).item()
    epoch_loss = 1.0*running_loss /num_batches
    epoch_acc = 1.0*running_corrects /num_batches

    return epoch_loss , epoch_acc , maintaining_F1

def run_epoch(train_iterator, val_iterator, epoch, num_batches, num_batches_val, optimizer):
	train_losses = []
	val_accuracies = []
	val_losses=[]
	losses = []

	titles = train_iterator[0]
	body = train_iterator[1]
	ans = train_iterator[2]

	self.train()
	for i in tqdm.trange(num_batches,file=sys.stdout, desc='Iterations'):
	    optimizer.zero_grad()
	    y_pred = self.calling([titles[0][i],titles[1][i]],[body[0][i],body[1][i]],[ans[0][i],ans[1][i]])
	    y = titles[2][i]
	    loss = self.loss(y_pred, y)
	    loss.backward()
	    losses.append(loss.detach().numpy())
	    optimizer.step()

	    if i % 10 == 0:
	        print("Iter: {},Epoch: {}\n".format(i+1,epoch))
	        avg_train_loss = np.mean(losses)
	        train_losses.append(avg_train_loss)
	        print("\tAverage training loss: {:.5f}\n".format(avg_train_loss))
	        losses = []
	        self.eval()
	        # Evalute Accuracy on validation set
	        with torch.no_grad():
	          val_loss , val_accuracy , curr_F1 = evaluate_model(val_iterator,num_batches_val,self.loss)
	          val_accuracies.append(val_accuracy)
	          val_losses.append(val_loss)
	          print("\tVal Accuracy: {:.4f}\n".format(val_accuracy))
	          print("\n")
	          self.train()
	        
	return train_losses, val_losses, val_accuracies , curr_F1


def train_model(model, batch_size, patience, n_epochs):

	np.random.seed(777)   # for reproducibility

	config = Config()

	dataset = Bilstm_Dataset()
	NUM_INSTANCES = dataset.__len__()
	NUM_INSTANCES = NUM_INSTANCES*config.sample
	TEST_RATIO = 0.3
	TEST_SIZE = int(NUM_INSTANCES * 0.3)

	indices = list(range(NUM_INSTANCES))

	test_idx = np.random.choice(indices, size = TEST_SIZE, replace = False)
	train_idx = list(set(indices) - set(test_idx))
	train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)

	train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
	test_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = test_sampler)


	best_val_loss = float("inf")

	loss_fn = nn.CrossEntropyLoss()
	model = BiLSTM(config, len(stack.word2index), embedding_matrix,loss_fn)
	optimizer = optim.Adam(model.parameters(), lr=config.lr)

	# initialize the early_stopping object
	early_stopping = EarlyStopping(patience=config.patience, verbose=True, delta=config.delta)

	if torch.cuda.is_available():
	    model.cuda()

	model.train()

	train_losses_plot = []
	val_accuracies_plot = []
	val_losses_plot = []
	confusion_matrix = []
	num_batches = len(train_iter[1][0]) #87
	num_batches_val = len(val_iter[1][0]) #37
	start_of_training = time.time()
	for i in range(config.epochs):
	    print ("Epoch: {}".format(i))
	    train_loss,val_losses,val_accuracy,curr_F1 = model.run_epoch(train_iter,val_iter,i,num_batches,num_batches_val,optimizer)
	    train_losses_plot.append(train_loss)
	    val_accuracies_plot.append(val_accuracy)
	    confusion_matrix.append(curr_F1)

	    early_stopping(val_losses[-1], model)
	        
	    if early_stopping.early_stop:
	        print("Early stopping....\n")
	        break

	    if np.mean(np.array(val_losses)) < best_val_loss:
	      best_val_loss = np.mean(np.array(val_losses))
	      best_model = model
	    
	end_of_training = time.time()

	print("\n\n Training Time : {:5.2f}".format(end_of_training-start_of_training))
	    
    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses


