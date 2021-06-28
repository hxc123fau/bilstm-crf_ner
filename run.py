from docopt import docopt
from vocab import Vocab
import time
import torch
import torch.nn as nn
import bilstm_crf
import utils
import random
from self_arg_parser import *
import new_bilstm_crf


def train():
    """ Training BiLSTMCRF model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args.sent_vocab_path)
    tag_vocab=Vocab.load(args.tag_vocab_path)
    print('sent_vocab',sent_vocab)
    train_filepath='./data/train.txt'
    train_data, validation_data = utils.generate_train_dev_dataset(train_filepath, sent_vocab, tag_vocab)
    print('num of training examples: %d' % (len(train_data)))
    print('num of validation examples: %d' % (len(validation_data)))

    max_epoch = int(args.max_epoch)
    log_step = int(args.log_step)
    validation_step=args.validation_step
    model_save_path = args.model_save_path
    optimizer_save_path = args.optimizer_save_path
    min_dev_loss = float('inf')
    device = args.device
    print('device',device)
    patience, decay_num = 0, 0

    model = bilstm_crf.BiLSTMCRF(sent_vocab, tag_vocab, float(args.dropout_rate), int(args.embedding_size),
                                 int(args.hidden_size)).to(device)
    vocab_size=len(sent_vocab)
    print('vocab_size',vocab_size)
    print('tag_size',len(tag_vocab))
    # model=new_bilstm_crf.BiLSTM_CRF(vocab_size, tag_vocab, args.embedding_size, args.hidden_size)
    model=model.to(device)
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         nn.init.normal_(param.data, 0, 0.01)
    #     else:
    #         nn.init.constant_(param.data, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    train_iter = 0  # train iter num
    record_loss_sum, record_tgt_word_sum, record_batch_size = 0, 0, 0  # sum in one training log
    cum_loss_sum, cum_tgt_word_sum, cum_batch_size = 0, 0, 0  # sum in one validation log
    record_start, cum_start = time.time(), time.time()

    print('start training...')
    for epoch in range(max_epoch):
        all_batch_sentences=utils.batch_iter(train_data, batch_size=int(args.batch_size))
        for sentences, tags in all_batch_sentences:
            train_iter += 1
            current_batch_size = len(sentences)
            print('pad_idx',sent_vocab[sent_vocab.PAD])
            print('111000',type(sentences),len(sentences),current_batch_size)
            sentences_idx, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            print('sent_lengths',sentences_idx.size(),sent_lengths)
            tags, _ = utils.pad(tags, tag_vocab[tag_vocab.PAD], device)
            print('tags',tags.size())

            # back propagation
            optimizer.zero_grad()
            # batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
            # print('sentences',sentences.size(),sentences)
            score, tag_seq=model(sentences_idx)
            batch_loss = model.loss_neg_log_likelihood(sentences_idx, tags)
            loss = batch_loss.mean()
            # Step 3. Run our forward pass.

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_max_norm))
            optimizer.step()

            record_loss_sum += batch_loss.sum().item()
            record_batch_size += current_batch_size
            record_tgt_word_sum += sum(sent_lengths)

            cum_loss_sum += batch_loss.sum().item()
            cum_batch_size += current_batch_size
            cum_tgt_word_sum += sum(sent_lengths)

            if train_iter % log_step == 0:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_tgt_word_sum / (time.time() - record_start),
                       record_loss_sum / record_batch_size, time.time() - record_start))
                record_loss_sum, record_batch_size, record_tgt_word_sum = 0, 0, 0
                # record_start = time.time()

            # if train_iter % validation_step == 0:
            #     print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
            #           (epoch + 1, train_iter, cum_tgt_word_sum / (time.time() - cum_start),
            #            cum_loss_sum / cum_batch_size, time.time() - cum_start))
            #     cum_loss_sum, cum_batch_size, cum_tgt_word_sum = 0, 0, 0

                # dev_loss = cal_dev_loss(model, dev_data, 64, sent_vocab, tag_vocab, device)

                # if dev_loss < min_dev_loss * float(args['--patience-threshold']):
                #     min_dev_loss = dev_loss
                #     model.save(model_save_path)
                #     torch.save(optimizer.state_dict(), optimizer_save_path)
                #     patience = 0
                # else:
                #     patience += 1
                #     if patience == int(args['--max-patience']):
                #         decay_num += 1
                #         if decay_num == int(args['--max-decay']):
                #             print('Early stop. Save result model to %s' % model_save_path)
                #             return
                #         lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                #         model = bilstm_crf.BiLSTMCRF.load(model_save_path, device)
                #         optimizer.load_state_dict(torch.load(optimizer_save_path))
                #         for param_group in optimizer.param_groups:
                #             param_group['lr'] = lr
                #         patience = 0
                # print('dev: epoch %d, iter %d, dev_loss %f, patience %d, decay_num %d' %
                #       (epoch + 1, train_iter, dev_loss, patience, decay_num))
                # cum_start = time.time()
                # if train_iter % log_every == 0:
                #     record_start = time.time()
    print('Reached %d epochs, Save result model to %s' % (max_epoch, model_save_path))



def cal_dev_loss(model, dev_data, batch_size, sent_vocab, tag_vocab, device):
    """ Calculate loss on the development data
    Args:
        model: the model being trained
        dev_data: development data
        batch_size: batch size
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        device: torch.device on which the model is trained
    Returns:
        the average loss on the dev data
    """
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sentences, tags in utils.batch_iter(dev_data, batch_size, shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[sent_vocab.PAD], device)
            batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences


if __name__ == '__main__':
    train()