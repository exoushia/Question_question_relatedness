# Question_question_relatedness

## Todo
* BERT
* update embeddings matrix instead creat new in test
* save predicted values for future reference
* Debug to see input to softmax layers
* cdb
* CNN
* tensorboard use
* reduce drop-out


## Usage
```
python main.py -h
```

You will get:

```
CNN text classificer

optional arguments:
  -h, --help            show this help message and exit
  -batch-size N         batch size for training [default: 50]
  -lr LR                initial learning rate [default: 0.01]
  -epochs N             number of epochs for train [default: 10]
  -dropout              the probability for dropout [default: 0.5]
  -max_norm MAX_NORM    l2 constraint of parameters
  -cpu                  disable the gpu
  -device DEVICE        device to use for iterate data
  -embed-dim EMBED_DIM
  -static               fix the embedding
  -kernel-sizes KERNEL_SIZES
                        Comma-separated kernel size to use for convolution
  -kernel-num KERNEL_NUM
                        number of each kind of kernel
  -class-num CLASS_NUM  number of class
  -shuffle              shuffle the data every epoch
  -num-workers NUM_WORKERS
                        how many subprocesses to use for data loading
                        [default: 0]
  -log-interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  -test-interval TEST_INTERVAL
                        how many epochs to wait before testing
  -save-interval SAVE_INTERVAL
                        how many epochs to wait before saving
  -predict PREDICT      predict the sentence given
  -snapshot SNAPSHOT    filename of model snapshot [default: None]
  -save-dir SAVE_DIR    where to save the checkpoint
```

## Reference
* paper here