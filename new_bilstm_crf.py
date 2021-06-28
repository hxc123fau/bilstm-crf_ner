# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from self_arg_parser import *

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    print('max_score',max_score)
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


START_TAG = "<START>"
STOP_TAG = "<END>"

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        print('self.word_embeds',self.word_embeds)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.batch_size=args.batch_size
        self.transitions = \
            nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)).to(self.device)
        # print('self.transitions',self.transitions.size())

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # START_TAG = "<START>"
        # STOP_TAG = "<END>"
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # self.hidden = self.init_hidden()
    @property
    def device(self):
        device_res=self.word_embeds.weight.device
        # print('device_res',device_res)
        return device_res

    def init_hidden(self,batch_size):
        init_hidden_state=(torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device),
                torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
        return init_hidden_state

    def _forward_alg(self, feats):
        batch_size=feats.size()[0]
        print('batch_size',batch_size,feats.size())
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full(( 1,self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0,self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.to(self.device)

        # Iterate through the sentence
        # for i in range(batch_size): # b,len,k
        sent_length = feats.size()[1]
        for i in range(sent_length):
            alphas_t = []  # The forward tensors at this time step
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                print('sss',feats[:,i,next_tag].size(),feats.size())
                emit_score = feats[:,i,next_tag].view(1, -1).expand(batch_size, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
            print('forward_var', forward_var.size())
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        print('lstm_sentence',sentence.size()) # b,len
        sentence_T=sentence.transpose(0,1) # len,b
        # embeds = self.word_embeds(sentence_T).view(len(sentence_T), 1, -1)
        print('sentence_T',sentence_T.shape)
        embeds = self.word_embeds(sentence_T)   # len,b,ed
        print('embeds',embeds.size())
        batch_size=embeds.size()[1]
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) #
        print('lstm_out',lstm_out.size()) # len,b,ed
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)  # len,b,k
        print('lstm_feats',lstm_feats.size())   # len,b,k
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        batch_score=[]
        batch_path=[]
        for sent_feat in feats: # feats (b,len,k)
            backpointers = []
            # print('tagset_size', self.tagset_size)
            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0
            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for word_feat in sent_feat:   #sentence
                bptrs_t = []  # holds the backpointers for this step
                viterbi_vars_t = []  # holds the viterbi variables for this step
                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    # print('forward_var',forward_var.size())
                    # print('transitions',self.transitions[next_tag].size())
                    next_tag_var = forward_var + self.transitions[next_tag].unsqueeze(0)
                    # print('next_tag_var',next_tag_var.size())
                    best_tag_id = argmax(next_tag_var)
                    # print('best_tag_id',best_tag_id)
                    bptrs_t.append(best_tag_id) # max prob of this tag in this word
                    viterbi_vars_t.append(next_tag_var[0][best_tag_id].view(1)) # next_tag_var (1,k)
                    # print('viterbi_vars_t',viterbi_vars_t)
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                # print('word_feat',word_feat.size())
                forward_var = (torch.cat(viterbi_vars_t) + word_feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path. 句子的最终路径
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()

            batch_path.append(best_path)
            batch_score.append(path_score)

        return batch_score, batch_path

    def loss_neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        feats_T = feats.transpose(0, 1)  # b,len,k
        print('feats_T',feats_T.size())
        forward_score = self._forward_alg(feats_T)
        gold_score = self._score_sentence(feats_T, tags)
        loss=forward_score - gold_score
        return loss

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        lstm_feats_T=lstm_feats.transpose(0,1) # b,len,k
        print('lstm_feats_T',type(lstm_feats_T),lstm_feats_T.size())
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats_T)
        return score, tag_seq

