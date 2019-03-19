
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 2, dropout=0)


class Model(nn.Module):
    def __init__(self, config, word_mat, char_mat):
        super().__init__()
        self.config = config
        # self.word_dim = config.glove_dim
        self.word_dim = 1024
        # self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        # self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        # self.word_emb.weight.requires_grad = False
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
        self.char_hidden = config.char_hidden
        self.hidden = config.hidden


        self.rnn = EncoderRNN(config.char_hidden+self.word_dim, config.hidden, 1, True, True, 1-config.keep_prob, False)
        self.qc_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_1 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.self_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_2 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_sp = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_sp = nn.Linear(config.hidden*2, 1)

        self.rnn_start = EncoderRNN(config.hidden+1, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_start = nn.Linear(config.hidden*2, 1)

        self.rnn_end = EncoderRNN(config.hidden*3+1, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_end = nn.Linear(config.hidden*2, 1)

        self.rnn_type = EncoderRNN(config.hidden*3+1, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_type = nn.Linear(config.hidden*2, 3)
        self.avgpool = torch.nn.AvgPool1d(kernel_size = 50)
        self.cache_S = 0

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False):
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (self.avgpool(context_idxs.float()).squeeze() > 0).float()
        # originally it was 2d, using a avgmask to restore the mask to 2d
        ques_mask = (self.avgpool(ques_idxs.float()).squeeze() > 0).float()

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

        context_word = elmo(context_idxs)['elmo_representations']
        context_word = (context_word[0] + context_word[1])

        # ques_word = self.word_emb(ques_idxs)
        ques_word =  elmo(ques_idxs)['elmo_representations']
        ques_word = (ques_word[0] + ques_word[1])


        context_output = torch.cat([context_word, context_ch], dim=2)
        print('after concatenation: ', context_output.size())
        ques_output = torch.cat([ques_word, ques_ch], dim=2)

        context_output = self.rnn(context_output, context_lens)
        ques_output = self.rnn(ques_output)

        output = self.qc_att(context_output, ques_output, ques_mask)
        output = self.linear_1(output)

        output_t = self.rnn_2(output, context_lens)
        output_t = self.self_att(output_t, output_t, context_mask)
        output_t = self.linear_2(output_t)

        output = output + output_t

        sp_output = self.rnn_sp(output, context_lens)

        start_output = torch.matmul(start_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,self.hidden:])
        end_output = torch.matmul(end_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,:self.hidden])
        sp_output = torch.cat([start_output, end_output], dim=-1)
        sp_output = self.linear_sp(sp_output)
        sp_output_aux = Variable(sp_output.data.new(sp_output.size(0), sp_output.size(1), 1).zero_())
        predict_support = torch.cat([sp_output_aux, sp_output], dim=-1).contiguous()

        sp_output = torch.matmul(all_mapping, sp_output)
        output = torch.cat([output, sp_output], dim=-1)

        output_start = self.rnn_start(output, context_lens)
        logit1 = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
        output_end = torch.cat([output, output_start], dim=2)
        output_end = self.rnn_end(output_end, context_lens)
        logit2 = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)

        output_type = torch.cat([output, output_end], dim=2)
        output_type = torch.max(self.rnn_type(output_type, context_lens), 1)[0]
        predict_type = self.linear_type(output_type)

        if not return_yp: return logit1, logit2, predict_type, predict_support

        outer = logit1[:,:,None] + logit2[:,None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return logit1, logit2, predict_type, predict_support, yp1, yp2

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
            print('Here is variable lens', type(lens), lens)
            print('input_lengths ', input_lengths)
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

# originally the shape of variables
# mask torch.Size([5, 23])
# input torch.Size([5, 1262, 160])
# memory torch.Size([5, 23, 160])
# input_dot torch.Size([5, 1262, 1])
# memory_dot torch.Size([5, 1, 23])
# corss_dot torch.Size([5, 1262, 23])
# att torch.Size([5, 1262, 1262])

# The shape of mask torch.Size([5, 23, 50])
# Inside the Biattention the input has the size of torch.Size([5, 1262, 160])
# Inside the Biattention the memory has the size of torch.Size([5, 23, 160])
# The shape of input_dot is torch.Size([5, 1262, 1])
# The shape of memory_dot is torch.Size([5, 1, 23])
# The shape of cross_dot is torch.Size([5, 1262, 23])
class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)
        print('The shape of mask', mask.shape)
        input = self.dropout(input)
        memory = self.dropout(memory)
        print('Inside the Biattention the input has the size of', input.size())
        print('Inside the Biattention the memory has the size of', memory.size())

        input_dot = self.input_linear(input)
        print('The shape of input_dot is', input_dot.size())
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        print('The shape of memory_dot is', memory_dot.size())
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        print('The shape of cross_dot is', cross_dot.size())
        att = input_dot + memory_dot + cross_dot
        print('the att shape', att.size())
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))
