import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import sys
import numpy as np

from Model.network_architecture import BiLSTM, EarlyStopping


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



# Create Model with specified optimizer and loss function
best_val_loss = float("inf")

config = Config()
loss_fn = nn.CrossEntropyLoss()
model = BiLSTM(config, len(stack.word2index), embedding_matrix,loss_fn)
optimizer = optim.Adam(model.parameters(), lr=config.lr)

'''
model.word_em.weight.data = train_dataset.fields["comment_text"].vocab.vectors
'''

# initialize the early_stopping object
patience = 5
delta = 0.01
early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta)

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

for i in tqdm.trange(config.epochs, file=sys.stdout, desc='Epochs'):
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