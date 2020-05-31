import tqdm
import torch
import sys

def evaluate_model(model, val_iterator,num_batches_val,loss_fn):
    all_y = []
    titles = val_iterator[0]
    body = val_iterator[1]
    ans = val_iterator[2]
    running_loss = 0
    running_corrects = 0
    maintaining_F1 = []
    for i in tqdm.trange(num_batches_val, file=sys.stdout, desc='Evaluation'):
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