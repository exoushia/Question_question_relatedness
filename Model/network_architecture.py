import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import evaluate

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
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

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), 'checkpoint.pt')
    #     self.val_loss_min = val_loss



class BiLSTM(nn.Module):

  def __init__(self, config, vocab_size, word_embeddings,loss_fn):
    super(BiLSTM, self).__init__()
    
    self.loss = loss_fn
    # self.optimizer = optimizer
    self.config = config
    self.dropout = self.config.dropout_keep
    
    #Layer 1: Word2Vec Embedding.
    self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
    self.embeddings.weight = nn.Parameter(torch.tensor(word_embeddings, dtype=torch.float32), requires_grad=False)

    # Layer 2: Bidirectional LSTM
    self.lstm = nn.LSTM(input_size= self.config.embed_size,
                            hidden_size=self.config.hidden_size,
                            num_layers=self.config.hidden_layers, 
                            dropout = self.config.dropout_keep,
                            bidirectional=True,
                            batch_first=True)
    
    # Layer 3: Attention Layer
    self.attn = nn.Dense(2*self.config.hidden_size, 1)
    
#     train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=False)

    # Layer 4: Rest of the layers
    self.net = nn.Sequential(nn.Linear(3, 50), nn.ReLU(), nn.Dropout(p=self.dropout), nn.Linear(50,self.config.output_size), nn.Softmax())
    # net.apply(init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
  

  def forward(self, pairs):
    x1 = self.embeddings(pairs[0][:])
    x2 = self.embeddings(pairs[1][:])
    lstm_out1, (h_n1, c_n1) = self.lstm(x1)
    lstm_out2, (h_n2, c_n2) = self.lstm(x2)

    #Concating both iterations of bilstm
    h_n1 = torch.cat([h_n1[0,:,:], h_n1[1,:,:]],-1).view(self.config.batch_size,256)
    h_n2 = torch.cat([h_n2[0,:,:], h_n2[1,:,:]],-1).view(self.config.batch_size,256)
    # print(h_n1.shape)

    weights = []
    for i in range(2*self.config.hidden_size):
      weights.append(self.attn(h_n1))
    normalized_weights = F.softmax(torch.cat(weights, 1), 1)
    # print("attn_applied")
    # print(normalized_weights.unsqueeze(0).shape) #1x32x256
    # print(h_n1.view(1, -1,2* self.config.hidden_size).shape) #1x32x256
    attn_applied = torch.bmm(normalized_weights.unsqueeze(0), h_n1.view(1, 2*self.config.hidden_size, -1))
    #If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) 
    # print(attn_applied.shape)   #1x32x32

    weights2 = []
    for i in range(2*self.config.hidden_size):
      weights2.append(self.attn(h_n2))
    normalized_weights2 = F.softmax(torch.cat(weights2, 1), 1)
    # print("attn_applied")
    # print(normalized_weights.unsqueeze(0).shape) #1x32x256
    # print(h_n1.view(1, -1,2* self.config.hidden_size).shape) #1x32x256
    attn_applied2 = torch.bmm(normalized_weights2.unsqueeze(0), h_n2.view(1, 2*self.config.hidden_size, -1))

    # print("shape\n")
    # print(lstm_out1.shape,lstm_out2.shape)
    
    #shape of hidden state - batch_size,hidden_dimension*2 -> dot product across second dimension
    # dotproduct = torch.tensor(np.sum(np.multiply(a.numpy(),b.numpy()),axis=1,keepdims=False),dtype=torch.float32)
    dotproduct = torch.tensor(np.sum(np.multiply(attn_applied.detach().numpy(), attn_applied2.detach().numpy()), axis=1),dtype=torch.float32).view(self.config.batch_size,-1)
    # print(dotproduct.shape)  #32x1
    return dotproduct
  
  def calling(self,t,b,a):
    inner_dot_titles = self.forward(t)
    inner_dot_body = self.forward(b)
    inner_dot_ans = self.forward(a)
    #need to concatenate these tensors along the right dimention - batch size
    concat_input_to_dense = torch.cat((inner_dot_titles,inner_dot_body,inner_dot_ans),1)
    #then dense layer
    #then activation
    # print("shape before applying view - concat_input_to_dense\n")
    # print(concat_input_to_dense.shape)

    # concat_input_to_dense = concat_input_to_dense.view(-1,)
    # print("\nshape after applying view - concat_input_to_dense\n")
    # print(concat_input_to_dense.shape)

    output = self.net(concat_input_to_dense)
    return output.view(-1,self.config.output_size)

  def run_epoch(self, train_iterator, val_iterator, epoch, num_batches, num_batches_val, optimizer):
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
              val_loss , val_accuracy , curr_F1 = evaluate_model(self,val_iterator,num_batches_val,self.loss)
              val_accuracies.append(val_accuracy)
              val_losses.append(val_loss)
              print("\tVal Accuracy: {:.4f}\n".format(val_accuracy))
              print("\n")
              self.train()
            
    return train_losses, val_losses, val_accuracies , curr_F1
