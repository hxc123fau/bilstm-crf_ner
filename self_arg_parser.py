"""
Options:
    --dropout-rate=<float>              dropout rate [default: 0.5]
    --embed-size=<int>                  size of word embedding [default: 256]
    --hidden-size=<int>                 size of hidden state [default: 256]
    --batch-size=<int>                  batch-size [default: 32]
    --max-epoch=<int>                   max epoch [default: 10]
    --clip_max_norm=<float>             clip max norm [default: 5.0]
    --lr=<float>                        learning rate [default: 0.001]
    --log-every=<int>                   log every [default: 10]
    --validation-every=<int>            validation every [default: 250]
    --patience-threshold=<float>        patience threshold [default: 0.98]
    --max-patience=<int>                time of continuous worse performance to decay lr [default: 4]
    --max-decay=<int>                   time of lr decay to early stop [default: 4]
    --lr-decay=<float>                  decay rate of lr [default: 0.5]
    --model-save-path=<file>            model save path [default: ./model/model.pth]
    --optimizer-save-path=<file>        optimizer save path [default: ./model/optimizer.pth]
    --cuda                              use GPU
"""
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--sent_vocab_path',default='./vocab/sent_vocab.json')
parser.add_argument('--tag_vocab_path',default='./vocab/tag_vocab.json')
parser.add_argument('--train_path',default='./data/train.txt')
parser.add_argument('--test_path',default='./data/test.txt')

parser.add_argument('--dropout_rate',type=float,default=0.5)
parser.add_argument('--embedding_size',type=int,default=256)
parser.add_argument('--hidden_size',type=int,default=256)
parser.add_argument('--batch_size',type=int,default=300)
parser.add_argument('--max_epoch',type=int,default=10)
parser.add_argument('--clip_max_norm', type=float,default= 5.0)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--validation_step', default=250)
parser.add_argument('--log_step',type=int,default=100)
parser.add_argument('--max_size',type=int,default=5000)
parser.add_argument('--cutoff_freq',type=int,default=2)

parser.add_argument('--model_save_path', default= './model/model.pth')
parser.add_argument('--optimizer_save_path', default='./model/optimizer.pth')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    args.device=device
else:
    device = torch.device('cpu')
    args.device=device
    args.device=device


