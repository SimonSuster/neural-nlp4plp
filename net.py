import os
import random
import re

from corpus_util import get_max_nsymb_batches

random.seed(0)

import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.metrics import accuracy_score, mean_squared_error

from util import TorchUtils, load_emb, f1_score, load_bert
from corpus_util import STAT, DYN
import traceback


class Encoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                 word_idx, pretrained_emb_path, bidir, bert_embs, feature_idx, feat_size, feat_padding_idx,
                 feat_emb_dim, feat_type, feat_onehot, cuda=0):
        super().__init__()
        self.n_lstm_layers = n_layers * 2 if bidir else n_layers
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.word_idx = word_idx
        self.pretrained_emb_path = pretrained_emb_path
        self.bidir = bidir
        self.bert_embs = bert_embs
        self.feature_idx = feature_idx
        self.feat_size = feat_size
        self.feat_padding_idx = feat_padding_idx
        self.feat_emb_dim = feat_emb_dim
        self.feat_type = feat_type
        self.feat_onehot = feat_onehot
        if self.feat_type:
            if self.feat_onehot:
                feat_dim = self.feat_size * len(self.feat_type)
            else:  # embedded
                feat_dim = self.feat_emb_dim * len(self.feat_type)
        else:
            feat_dim = 0
        self.feat_dim = feat_dim
        self.final_emb_dim = self.emb_dim + self.feat_dim

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        self.hidden_in = self.init_hidden()  # initialize cell states

        if pretrained_emb_path is not None:
            self.word_embeddings, dim = load_emb(pretrained_emb_path, word_idx, freeze=False)
            assert dim == self.emb_dim
        elif bert_embs:
            self.word_embeddings, dim = load_bert(bert_embs, freeze=False)
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim,
                                                padding_idx=padding_idx)  # embedding layer, initialized at random

        if feature_idx is not None:
            if self.feat_onehot:
                self.feat_embeddings = self.one_hot
            else:
                self.feat_embeddings = nn.Embedding(self.feat_size, self.feat_emb_dim, padding_idx=feat_padding_idx)

        self.lstm = nn.LSTM(self.final_emb_dim, self.hidden_dim, num_layers=n_layers,
                            dropout=self.dropout, bidirectional=self.bidir)  # lstm layers
        self.to(self.device)

    def init_hidden(self):
        '''
        initializes hidden and cell states to zero for the first input
        '''
        h0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).to(self.device)

        return (h0, c0)

    def one_hot(self, features):
        """
        :param features: b * max_seq_len * len_feat_type
        :return: b * max_seq_len * len_feat_type * feat_size
        """
        idx_first = 0
        idx_last = self.feat_size
        x = torch.randint(idx_first, idx_last, (features.size(0), features.size(1), features.size(2), 1),
                          dtype=torch.long)
        z = torch.zeros(x.size(0), x.size(1), x.size(2), self.feat_size)
        z.scatter_(3, x, 1)

        return z

    def forward(self, sentence, features, sent_lengths, hidden):
        sort, unsort = TorchUtils.get_sort_unsort(sent_lengths)
        embs = self.word_embeddings(sentence).to(self.device)  # word sequence to embedding sequence
        if features is not None:
            # concatenate for all feat types
            feat_embs = torch.cat([f_t for f_t in self.feat_embeddings(features).permute(1, 0, 2, 3)], dim=2) \
                # match dims with embs
            feat_embs = feat_embs.permute(1, 0, 2).to(self.device)
            embs = torch.cat([embs, feat_embs], dim=2).to(self.device)

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = (hidden[0][:, :cur_batch_len, :].contiguous(), hidden[1][:, :cur_batch_len, :].contiguous())

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)

        # lstm_out: output of last lstm layer after every time step
        # hidden gets updated and cell states at the end of the sequence
        lstm_out, hidden = self.lstm(embs, hidden)
        # pad the sequences again to convert to original padded data shape
        # traceback.print_stack(limit=4)
        # print(embs[0].shape,embs[1].shape, lstm_out[0].shape, lstm_out[1].shape, hidden[0].shape, hidden[1].shape)

        lstm_out, lengths = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=False)

        # unsort batch
        lstm_out = lstm_out[:, unsort]
        hidden = (hidden[0][:, unsort, :], hidden[1][:, unsort, :])

        return lstm_out, hidden


class LSTMClassifier(nn.Module):
    # based on https://github.com/MadhumitaSushil/sepsis/blob/master/src/classifiers/lstm.py
    def __init__(self, n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, label_size, batch_size,
                 word_idx, pretrained_emb_path, f_model, cuda=0):
        super().__init__()
        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_labels = label_size
        self.word_idx = word_idx
        self.pretrained_emb_path = pretrained_emb_path
        self.f_model = f_model

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        self.encoder = Encoder(n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                               word_idx, pretrained_emb_path, cuda=cuda)
        self.hidden2label = nn.Linear(self.hidden_dim, self.n_labels)  # hidden to output layer
        self.to(self.device)

    def forward(self, sentence, sent_lengths):
        hidden = self.encoder.init_hidden()
        lstm_output, hidden = self.encoder(sentence, sent_lengths, hidden)
        # use the output of the last LSTM layer at the end of the last valid timestep to predict output
        # If sequence len is constant, using hidden[0] is the same as lstm_out[-1].
        # For variable len seq, use hidden[0] for the hidden state at last valid timestep. Do it for the last hidden layer
        y = self.hidden2label(hidden[0][-1])
        y = F.log_softmax(y, dim=1)

        return y

    def loss(self, fwd_out, target):
        # NLL loss to be used when logits have log-softmax output.
        # If softmax layer is not added, directly CrossEntropyLoss can be used.
        loss_fn = nn.NLLLoss()
        return loss_fn(fwd_out, target)

    def train_model(self, corpus, dev_corpus, corpus_encoder, feature_encoder, n_epochs, optimizer):

        self.train()

        optimizer = optimizer
        best_acc = 0.

        for i in range(n_epochs):
            running_loss = 0.0

            # shuffle the corpus
            corpus.shuffle()
            # get train batch
            for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, self.batch_size)):
                cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)

                # forward pass
                fwd_out = self.forward(cur_insts, cur_lengths)

                # loss calculation
                loss = self.loss(fwd_out, cur_labels)

                # backprop
                optimizer.zero_grad()  # reset tensor gradients
                loss.backward()  # compute gradients for network params w.r.t loss
                optimizer.step()  # perform the gradient update step
                running_loss += loss.item()
            y_pred, y_true = self.predict(dev_corpus, corpus_encoder)
            self.train()  # set back the train mode
            dev_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            if i == 0 or dev_acc > best_acc:
                self.save(self.f_model)
                best_acc = dev_acc
            print('ep %d, loss: %.3f, dev_acc: %.3f' % (i, running_loss, dev_acc))

    def predict(self, corpus, corpus_encoder):
        self.eval()
        y_pred = list()
        y_true = list()

        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, self.batch_size)):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)
            y_true.extend(cur_labels.cpu().numpy())

            # forward pass
            fwd_out = self.forward(cur_insts, cur_lengths)
            __, cur_preds = torch.max(fwd_out.detach(), 1)  # first return value is the max value, second is argmax
            y_pred.extend(cur_preds.cpu().numpy())

        return y_pred, y_true

    def save(self, f_model='lstm_classifier.tar', dir_model='../out/'):

        net_params = {'n_layers': self.n_lstm_layers,
                      'hidden_dim': self.hidden_dim,
                      'vocab_size': self.vocab_size,
                      'padding_idx': self.encoder.word_embeddings.padding_idx,
                      'embedding_dim': self.emb_dim,
                      'dropout': self.dropout,
                      'label_size': self.n_labels,
                      'batch_size': self.batch_size,
                      'word_idx': self.word_idx,
                      'pretrained_emb_path': self.pretrained_emb_path,
                      'f_model': self.f_model
                      }

        # save model state
        state = {
            'net_params': net_params,
            'state_dict': self.state_dict(),
        }

        TorchUtils.save_model(state, f_model, dir_model)

    @classmethod
    def load(cls, f_model='lstm_classifier.tar', dir_model='../out/'):

        state = TorchUtils.load_model(f_model, dir_model)
        classifier = cls(**state['net_params'])
        classifier.load_state_dict(state['state_dict'])

        return classifier


class LSTMRegression(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size, word_idx,
                 pretrained_emb_path, f_model, cuda=0):
        super().__init__()

        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.word_idx = word_idx
        self.pretrained_emb_path = pretrained_emb_path
        self.f_model = f_model

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        if pretrained_emb_path is not None:
            self.word_embeddings, dim = load_emb(pretrained_emb_path, word_idx, freeze=False)
            assert dim == self.emb_dim
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim,
                                                padding_idx=padding_idx)  # embedding layer, initialized at random

        self.encoder = Encoder(n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                               word_idx, pretrained_emb_path, cuda=cuda)

        self.hidden2label = nn.Linear(self.hidden_dim, 1)  # hidden to output node
        # self.sigmoid = nn.Sigmoid()
        self.to(self.device)

    def forward(self, sentence, sent_lengths):
        hidden = self.encoder.init_hidden()
        lstm_output, hidden = self.encoder(sentence, sent_lengths, hidden)
        # use the output of the last LSTM layer at the end of the last valid timestep to predict output
        # If sequence len is constant, using hidden[0] is the same as lstm_out[-1].
        # For variable len seq, use hidden[0] for the hidden state at last valid timestep. Do it for the last hidden layer
        y = self.hidden2label(hidden[0][-1])
        # y = self.sigmoid(y)
        return y

    def loss(self, fwd_out, target):
        loss_fn = nn.MSELoss()
        return loss_fn(fwd_out, target)

    def train_model(self, corpus, dev_corpus, corpus_encoder, feature_encoder, n_epochs, optimizer):

        self.train()

        optimizer = optimizer
        best_mse = np.inf

        for i in range(n_epochs):
            running_loss = 0.0

            # shuffle the corpus
            corpus.shuffle()
            # get train batch
            for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, self.batch_size)):
                cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)

                # forward pass
                fwd_out = self.forward(cur_insts, cur_lengths)

                # loss calculation
                loss = self.loss(fwd_out, cur_labels.unsqueeze(1))

                # backprop
                optimizer.zero_grad()  # reset tensor gradients
                loss.backward()  # compute gradients for network params w.r.t loss
                optimizer.step()  # perform the gradient update step
                running_loss += loss.item()
            y_pred, y_true = self.predict(dev_corpus, corpus_encoder)
            self.train()  # set back the train mode
            dev_mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            if i == 0 or dev_mse < best_mse:
                self.save(self.f_model)
                best_mse = dev_mse
            print('ep %d, loss: %.3f, dev_mse: %.3f' % (i, running_loss, dev_mse))

    def predict(self, corpus, corpus_encoder):
        self.eval()
        y_pred = list()
        y_true = list()

        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, self.batch_size)):
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)
            y_true.extend(cur_labels.cpu().numpy())

            # forward pass
            fwd_out = self.forward(cur_insts, cur_lengths)
            cur_preds = fwd_out.detach()
            y_pred.extend(cur_preds.cpu().numpy())

        return y_pred, y_true

    def save(self, f_model='lstm_regression.tar', dir_model='../out/'):

        net_params = {'n_layers': self.n_lstm_layers,
                      'hidden_dim': self.hidden_dim,
                      'vocab_size': self.vocab_size,
                      'padding_idx': self.encoder.word_embeddings.padding_idx,
                      'embedding_dim': self.emb_dim,
                      'dropout': self.dropout,
                      'batch_size': self.batch_size,
                      'word_idx': self.word_idx,
                      'pretrained_emb_path': self.pretrained_emb_path,
                      'f_model': self.f_model
                      }

        # save model state
        state = {
            'net_params': net_params,
            'state_dict': self.state_dict(),
        }

        TorchUtils.save_model(state, f_model, dir_model)

    @classmethod
    def load(cls, f_model='lstm_regression.tar', dir_model='../out/'):

        state = TorchUtils.load_model(f_model, dir_model)
        classifier = cls(**state['net_params'])
        classifier.load_state_dict(state['state_dict'])

        return classifier


class PointerAttention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim, cuda=0):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)
        self.to(self.device)

    def forward(self, input,
                context,
                mask=None):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if mask is not None and len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class PointerDecoder(nn.Module):
    """
    Decoder model for Pointer-Net (based on https://github.com/shirgur/PointerNet.git)
    """

    def __init__(self, hidden_dim, vocab_size, padding_idx, embedding_dim, word_idx, pretrained_emb_path, output_len,
                 feature_idx, feat_size, feat_padding_idx, feat_emb_dim, cuda=0):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        self.emb_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_len = output_len
        self.feature_idx = feature_idx
        self.feat_size = feat_size
        self.feat_padding_idx = feat_padding_idx
        self.feat_emb_dim = feat_emb_dim
        self.final_emb_dim = self.emb_dim + (self.feat_emb_dim if self.feat_emb_dim is not None else 0)

        if pretrained_emb_path is not None:
            self.word_embeddings, dim = load_emb(pretrained_emb_path, word_idx, freeze=False)
            assert dim == self.emb_dim
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim,
                                                padding_idx=padding_idx)  # embedding layer, initialized at random
        if feature_idx is not None:
            self.feat_embeddings = nn.Embedding(feat_size, feat_emb_dim, padding_idx=feat_padding_idx)

        self.input_to_hidden = nn.Linear(self.final_emb_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = PointerAttention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)
        self.to(self.device)

    def forward(self, sentence, features, sent_lengths, decoder_input, hidden, context):
        """
        Decoder - Forward-pass

        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """
        # sort, unsort = TorchUtils.get_sort_unsort(sent_lengths)
        embs = self.word_embeddings(sentence).to(self.device)  # word sequence to embedding sequence
        embs = embs.permute(1, 0, 2)
        if features is not None:
            feat_embs = self.feat_embeddings(features).to(self.device)  # feature sequence to embedding sequence
            feat_embs = feat_embs.permute(1, 0, 2)
            embs = torch.cat([embs, feat_embs], dim=2)

        input_length = embs.size(1)

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = (hidden[0][:cur_batch_len, :], hidden[1][:cur_batch_len, :])

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        # embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(cur_batch_len, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(cur_batch_len, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(self.output_len):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            # to conceal previous max probs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask = mask * (1 - one_hot_pointers)  # constrained to always be different?

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.final_emb_dim).byte()
            decoder_input = embs[embedding_mask.data].view(cur_batch_len, self.final_emb_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)  # (b * output_len * input_len)
        pointers = torch.cat(pointers, 1)  # (b * output_len)

        return (outputs, pointers), hidden


class Decoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, padding_idx, label_padding_idx, embedding_dim, word_idx,
                 pretrained_emb_path, max_output_len, n_labels, cuda=0):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        self.emb_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_output_len = max_output_len
        self.n_labels = n_labels

        self.teacher_forcing = True
        self.label_embeddings = nn.Embedding(self.n_labels, self.emb_dim,
                                             padding_idx=label_padding_idx)  # embedding layer, initialized at random
        # self.input_to_hidden = nn.Linear(self.final_emb_dim, 4 * hidden_dim)
        self.input_to_hidden = nn.Linear(self.emb_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = PointerAttention(hidden_dim, hidden_dim)
        # self.out = nn.Linear(hidden_dim + hidden_dim + self.final_emb_dim, self.n_labels)
        self.out = nn.Linear(hidden_dim + hidden_dim + self.emb_dim, self.n_labels)

        # Used for propagating .cuda() command
        self.to(self.device)

    def forward(self, sent_lengths, output_length, decoder_input, hidden, context, cur_labels):
        """
        Decoder - Forward-pass

        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = (hidden[0][:cur_batch_len, :], hidden[1][:cur_batch_len, :])

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        # embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)
        outputs = []
        preds = []

        is_inference = output_length is None
        if is_inference:
            # predict for max len so that we can use batches, prune later
            output_length = self.max_output_len
        # Recurrence loop
        for i in range(output_length):
            h_t, c_t, outs = self.step(i, decoder_input, hidden, context)
            hidden = (h_t, c_t)
            # Get maximum probabilities and indices
            max_probs, indices = outs.max(1)
            # Embed output labels for next input
            if self.teacher_forcing and not is_inference:
                decoder_input = self.label_embeddings(cur_labels[:, i])
            else:
                decoder_input = self.label_embeddings(indices)

            outputs.append(outs.unsqueeze(0))
            preds.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)  # (b * output_len * n_labels)
        preds = torch.cat(preds, 1)  # (b * output_len)

        return (outputs, preds), hidden

    def step(self, i, x, hidden, context):
        """
        Recurrence step function

        :param int i: time step
        :param Tensor x: Input at time t
        :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
        :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
        """

        # Regular LSTM
        h, c = hidden

        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        input, forget, cell, out = gates.chunk(4, 1)

        input = F.sigmoid(input)
        forget = F.sigmoid(forget)
        cell = F.tanh(cell)
        out = F.sigmoid(out)

        c_t = (forget * c) + (input * cell)
        h_t = out * F.tanh(c_t)

        # Attention section
        weighted, output_att = self.att(h_t, context, None)
        hidden_t = F.tanh(self.hidden_out(torch.cat((weighted, h_t), 1)))

        # hidden_t: b*hidden_dim
        # weighted: b*hidden_dim
        # x: b*final_emb_dim
        output = self.out(torch.cat((hidden_t, weighted, x), dim=1))

        return hidden_t, c_t, output


class ConstrainedDecoderForSingleDec(nn.Module):
    def __init__(self, hidden_dim, vocab_size, padding_idx, label_padding_idx, embedding_dim, word_idx,
                 pretrained_emb_path, max_output_len, n_labels, label_idx, constrained_decoding, cuda=0):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        self.emb_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_output_len = max_output_len
        self.n_labels = n_labels
        self.label_idx = label_idx
        self.constrained_decoding = constrained_decoding
        if self.constrained_decoding is not None and "mod5" in constrained_decoding:
            self.par_idxs = [v for k,v in label_idx.items() if k[:-1] in STAT | DYN]
        self.teacher_forcing = True
        self.label_embeddings = nn.Embedding(self.n_labels, self.emb_dim,
                                             padding_idx=label_padding_idx)  # embedding layer, initialized at random
        # self.input_to_hidden = nn.Linear(self.final_emb_dim, 4 * hidden_dim)
        self.input_to_hidden = nn.Linear(self.emb_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = PointerAttention(hidden_dim, hidden_dim)
        # self.out = nn.Linear(hidden_dim + hidden_dim + self.final_emb_dim, self.n_labels)
        self.out_dim = hidden_dim + hidden_dim + self.emb_dim
        if self.constrained_decoding is not None and "mod5" in self.constrained_decoding:
            self.out_dim += self.emb_dim + hidden_dim
        self.out = nn.Linear(self.out_dim, self.n_labels)

        # Used for propagating .cuda() command
        self.to(self.device)

    def forward(self, sent_lengths, output_length, decoder_input, hidden, context, cur_labels, cur_max_nsymbs):
        """
        Decoder - Forward-pass

        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = (hidden[0][:cur_batch_len, :], hidden[1][:cur_batch_len, :])

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        # embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)
        outputs = []
        preds = []

        if self.constrained_decoding is not None and "mod5" in self.constrained_decoding:
            par_decoder_input = decoder_input.clone()  # will hold parent label embeddings
            par_hidden = hidden[0].clone()  # will hold parent hidden state

        is_inference = output_length is None
        if is_inference:
            # predict for max len so that we can use batches, prune later
            output_length = self.max_output_len
        # Recurrence loop
        for i in range(output_length):
            if self.constrained_decoding is not None and "mod5" in self.constrained_decoding:
                h_t, c_t, outs = self.step(i, decoder_input, hidden, context, par_decoder_input, par_hidden)
            else:
                h_t, c_t, outs = self.step(i, decoder_input, hidden, context)
            if self.constrained_decoding is not None and "mod4" in self.constrained_decoding:
                outs = self.max_nsymb_mask(outs, cur_max_nsymbs)
            hidden = (h_t, c_t)
            # Get maximum probabilities and indices
            max_probs, indices = outs.max(1)
            # Embed output labels for next input
            if self.teacher_forcing and not is_inference:
                decoder_input = self.label_embeddings(cur_labels[:, i])
            else:
                decoder_input = self.label_embeddings(indices)

            if self.constrained_decoding is not None and "mod5" in self.constrained_decoding:
                idxs = cur_labels[:, i] if self.teacher_forcing and not is_inference else indices
                for c, l in enumerate(idxs):  # check if label is a parent label
                    if l in self.par_idxs:
                        par_decoder_input[c] = decoder_input[c]  # update if parent
                        par_hidden[c] = hidden[0][c]  # update if parent

            outputs.append(outs.unsqueeze(0))
            preds.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)  # (b * output_len * n_labels)
        preds = torch.cat(preds, 1)  # (b * output_len)

        return (outputs, preds), hidden

    def step(self, i, x, hidden, context, par_decoder_input=None, par_hidden=None):
        """
        Recurrence step function

        :param int i: time step
        :param Tensor x: Input at time t
        :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
        :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
        """

        # Regular LSTM
        h, c = hidden

        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        input, forget, cell, out = gates.chunk(4, 1)

        input = F.sigmoid(input)
        forget = F.sigmoid(forget)
        cell = F.tanh(cell)
        out = F.sigmoid(out)

        c_t = (forget * c) + (input * cell)
        h_t = out * F.tanh(c_t)

        # Attention section
        weighted, output_att = self.att(h_t, context, None)
        hidden_t = F.tanh(self.hidden_out(torch.cat((weighted, h_t), 1)))

        # hidden_t: b*hidden_dim
        # weighted: b*hidden_dim
        # x: b*final_emb_dim
        if self.constrained_decoding is not None and "mod5" in self.constrained_decoding:
            output = self.out(torch.cat((hidden_t, weighted, x, par_decoder_input, par_hidden), dim=1))
        else:
            output = self.out(torch.cat((hidden_t, weighted, x), dim=1))

        return hidden_t, c_t, output

    def max_nsymb_mask(self, outs, cur_max_nsymbs):
        mask = torch.ones_like(outs, device=self.device)
        for b, n_max in enumerate(cur_max_nsymbs):
            for k, v in self.label_idx.items():
                ns = re.findall("^n(\d+)$", k)
                if not ns:
                    continue
                else:
                    n = ns.pop()
                    if n_max is None or int(n) > n_max:
                        mask[b, v] = 0.
        return outs * mask


class ConstrainedDecoderForSplitDec(nn.Module):
    def __init__(self, hidden_dim, vocab_size, padding_idx, label_padding_idx, embedding_dim, word_idx,
                 pretrained_emb_path, max_output_len, n_labels, n_labels_dec1, label_idx_dec1, label_idx_dec2,
                 constrained_decoding, cuda=0):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        :param list constrained_decoding: List of modifications (features and constraints) to use:
               mod1: label_dec1 is embedded using label_embeddings, label_dec1 as input to LSTM_dec2
               mod2: label_dec1 is represented as output distribution over all labels of dec1, label_dec1 as input to output layer of dec2
               mod3: masking for types (numbers) on output of dec2
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')

        self.emb_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_output_len = max_output_len
        self.n_labels = n_labels
        self.n_labels_dec1 = n_labels_dec1
        self.inv_label_idx_dec1 = {v: k for k, v in label_idx_dec1.items()}
        self.label_idx_dec2 = label_idx_dec2
        # for masking numbers
        self.num_labels_dec2 = [v for k, v in self.label_idx_dec2.items() if re.findall("\d+", k)]
        self.nonnum_labels_dec2 = list(set(self.label_idx_dec2.values()) - set(self.num_labels_dec2))
        self.constrained_decoding = constrained_decoding
        self.teacher_forcing = True
        self.label_embeddings = nn.Embedding(self.n_labels, self.emb_dim,
                                             padding_idx=label_padding_idx)  # embedding layer, initialized at random
        if self.constrained_decoding is not None and "mod1" in self.constrained_decoding:
            # modification 1: dec1 feat for hidden in split decoder
            self.input_to_hidden = nn.Linear(2 * self.emb_dim, 4 * hidden_dim)
        else:
            self.input_to_hidden = nn.Linear(self.final_emb_dim, 4 * hidden_dim)
        # self.input_to_hidden = nn.Linear(self.emb_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = PointerAttention(hidden_dim, hidden_dim)
        if self.constrained_decoding is not None and "mod2" in self.constrained_decoding:
            # modification 2: dec1 feat for output: in split decoder
            self.out = nn.Linear(hidden_dim + hidden_dim + self.emb_dim + self.n_labels_dec1, self.n_labels)
        else:
            self.out = nn.Linear(hidden_dim + hidden_dim + self.emb_dim, self.n_labels)

        # Used for propagating .cuda() command
        self.to(self.device)

    #    def __init__(self, *args, **kwargs):
    #        """
    #        Initiate Decoder
    #
    #        :param int embedding_dim: Number of embeddings in Pointer-Net
    #        :param int hidden_dim: Number of hidden units for the decoder's RNN
    #        """
    #        super().__init__(*args, **kwargs)
    #        self.input_to_hidden = nn.Linear(2 * self.emb_dim, 4 * self.hidden_dim)

    def forward(self, sent_lengths, output_length, decoder_input, hidden, context, cur_labels, labels=None,
                label_embeddings=None, dec1_outputs=None):
        """
        Decoder - Forward-pass

        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = (hidden[0][:cur_batch_len, :], hidden[1][:cur_batch_len, :])

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        # embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)
        outputs = []
        preds = []

        is_inference = output_length is None
        if is_inference:
            # predict for max len so that we can use batches, prune later
            output_length = self.max_output_len
        # Recurrence loop
        for i in range(output_length):
            h_t, c_t, outs = self.step(i, decoder_input, hidden, context, labels, label_embeddings, dec1_outputs,
                                       cur_batch_len)
            hidden = (h_t, c_t)
            # Get maximum probabilities and indices
            max_probs, indices = outs.max(1)
            # Embed output labels for next input
            if self.teacher_forcing and not is_inference:
                decoder_input = self.label_embeddings(cur_labels[:, i])
            else:
                decoder_input = self.label_embeddings(indices)

            outputs.append(outs.unsqueeze(0))
            preds.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)  # (b * output_len * n_labels)
        preds = torch.cat(preds, 1)  # (b * output_len)

        return (outputs, preds), hidden

    def step(self, i, x, hidden, context, labels, label_embeddings, dec1_outputs, cur_batch_len):
        """
        Recurrence step function

        :param int i: time step
        :param Tensor x: Input at time t
        :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
        :param Tensor label: label from dec1 at time t
        :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
        """

        # Regular LSTM
        h, c = hidden

        # modification 1: dec1 feat for hidden in split decoder
        if self.constrained_decoding is not None and "mod1" in self.constrained_decoding:
            gates = self.input_to_hidden(torch.cat((x, label_embeddings(labels[:, i])), dim=1)) + self.hidden_to_hidden(
                h)
        else:
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        input, forget, cell, out = gates.chunk(4, 1)

        input = F.sigmoid(input)
        forget = F.sigmoid(forget)
        cell = F.tanh(cell)
        out = F.sigmoid(out)

        c_t = (forget * c) + (input * cell)
        h_t = out * F.tanh(c_t)

        # Attention section
        weighted, output_att = self.att(h_t, context, None)
        hidden_t = F.tanh(self.hidden_out(torch.cat((weighted, h_t), 1)))

        # hidden_t: b*hidden_dim
        # weighted: b*hidden_dim
        # x: b*final_emb_dim

        if self.constrained_decoding is not None and "mod2" in self.constrained_decoding:
            # modification 2: dec1 feat for output in split decoder
            output = self.out(torch.cat((hidden_t, weighted, x, dec1_outputs[:, i, :]), dim=1))
        else:
            output = self.out(torch.cat((hidden_t, weighted, x), dim=1))

        if self.constrained_decoding is not None and "mod3" in self.constrained_decoding:
            # modification 3: masking for numbers in split decoder
            output = output * self.number_mask(i, labels, cur_batch_len)

        return hidden_t, c_t, output

    def number_mask(self, i, labels, cur_batch_len):
        # build a mask based on dec1 labels (types)
        # if label is l or n, dec2 should produce a number, else not a number
        to_num_mask = []  # b
        for k in labels[:, i]:
            to_num_mask.append(self.inv_label_idx_dec1[k.item()] in {"l", "n"})
        mask1 = torch.zeros(cur_batch_len, self.n_labels).to(self.device)
        for b, num in enumerate(to_num_mask):
            if num:
                mask1[b, self.num_labels_dec2] = 1.
            else:
                mask1[b, self.nonnum_labels_dec2] = 1.

        return mask1


class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self,
                 n_layers,
                 hidden_dim,
                 vocab_size,
                 padding_idx,
                 embedding_dim,
                 dropout,
                 batch_size,
                 word_idx,
                 pretrained_emb_path,
                 output_len,
                 f_model,
                 bidir=False,
                 feature_idx=None,
                 feat_size=None,
                 feat_padding_idx=None,
                 feat_emb_dim=None,
                 cuda=0):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')
        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.word_idx = word_idx
        self.pretrained_emb_path = pretrained_emb_path
        self.output_len = output_len
        # decoder output length
        self.f_model = f_model
        if bidir:
            raise NotImplementedError
        self.bidir = bidir
        self.feature_idx = feature_idx
        self.feat_size = feat_size
        self.feat_padding_idx = feat_padding_idx
        self.feat_emb_dim = feat_emb_dim
        self.final_emb_dim = self.emb_dim + (self.feat_emb_dim if self.feat_emb_dim is not None else 0)

        self.encoder = Encoder(n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                               word_idx, pretrained_emb_path, feature_idx, feat_size, feat_padding_idx,
                               feat_emb_dim, cuda=cuda)
        self.decoder = PointerDecoder(hidden_dim, vocab_size, padding_idx, embedding_dim, word_idx, pretrained_emb_path,
                                      output_len, feature_idx, feat_size, feat_padding_idx, feat_emb_dim, cuda=cuda)
        self.decoder_input0 = Parameter(torch.FloatTensor(self.final_emb_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform(self.decoder_input0, -1, 1)
        self.to(self.device)

    #    def forward(self, inputs):
    def forward(self, sentence, features, sent_lengths):
        # input_length = inputs.size(1)
        cur_batch_len = len(sent_lengths)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(cur_batch_len, -1)

        # inputs = inputs.view(batch_size * input_length, -1)
        # embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        enc_hidden0 = self.encoder.init_hidden()
        # encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, enc_hidden0)
        encoder_outputs, encoder_hidden = self.encoder(sentence, features, sent_lengths, enc_hidden0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        if self.bidir:
            decoder_hidden0 = (torch.cat(encoder_hidden[0][-2:], dim=-1),
                               torch.cat(encoder_hidden[1][-2:], dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(sentence, features, sent_lengths,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return outputs, pointers

    def loss(self, fwd_out, target):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(fwd_out, target)

    def train_model(self, corpus, dev_corpus, corpus_encoder, feature_encoder, n_epochs, optimizer):

        self.train()

        optimizer = optimizer
        best_acc = 0.

        for i in range(n_epochs):
            running_loss = 0.0

            # shuffle the corpus
            corpus.shuffle()
            # potential external features
            if feature_encoder is not None:
                _features = feature_encoder.get_feature_batches(corpus, self.batch_size)
            else:
                _features = None
            # get train batch
            for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, self.batch_size)):
                cur_feats = _features.__next__() if _features is not None else None
                if _features is not None:
                    assert len(cur_feats) == len(cur_insts)
                    cur_feats, cur_feat_lengths = feature_encoder.feature_batch_to_tensors(cur_feats, self.device)
                cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)
                cur_labels = cur_labels.long()
                # forward pass
                fwd_out, pointers = self.forward(cur_insts, cur_feats, cur_lengths)
                fwd_out = fwd_out.contiguous().view(-1, fwd_out.size()[-1])
                # loss calculation
                loss = self.loss(fwd_out, cur_labels.view(-1).long())

                # backprop
                optimizer.zero_grad()  # reset tensor gradients
                loss.backward()  # compute gradients for network params w.r.t loss
                optimizer.step()  # perform the gradient update step
                running_loss += loss.item()
            y_pred, y_true = self.predict(dev_corpus, feature_encoder, corpus_encoder)
            if self.output_len > 1:
                y_true = [str(y) for y in y_true]
                y_pred = [str(y) for y in y_pred]
            self.train()  # set back the train mode
            dev_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            if i == 0 or dev_acc > best_acc:
                self.save(self.f_model)
                best_acc = dev_acc
            print('ep %d, loss: %.3f, dev_acc: %.3f' % (i, running_loss, dev_acc))

    def predict(self, corpus, feature_encoder, corpus_encoder):
        self.eval()
        y_pred = list()
        y_true = list()

        # potential external features
        if feature_encoder is not None:
            _features = feature_encoder.get_feature_batches(corpus, self.batch_size)
        else:
            _features = None
        for idx, (cur_insts, cur_labels) in enumerate(corpus_encoder.get_batches(corpus, self.batch_size)):
            cur_feats = _features.__next__() if _features is not None else None
            if _features is not None:
                assert len(cur_feats) == len(cur_insts)
                cur_feats, cur_feat_lengths = feature_encoder.feature_batch_to_tensors(cur_feats, self.device)
            cur_insts, cur_labels, cur_lengths = corpus_encoder.batch_to_tensors(cur_insts, cur_labels, self.device)
            y_true.extend(cur_labels.cpu().numpy())

            # forward pass
            _, pointers = self.forward(cur_insts, cur_feats, cur_lengths)

            y_pred.extend(pointers.squeeze(1).cpu().numpy())

        return y_pred, y_true

    def save(self, f_model='lstm_pointer.tar', dir_model='../out/'):

        net_params = {'n_layers': self.n_lstm_layers,
                      'hidden_dim': self.hidden_dim,
                      'vocab_size': self.vocab_size,
                      'padding_idx': self.encoder.word_embeddings.padding_idx,
                      'embedding_dim': self.emb_dim,
                      'dropout': self.dropout,
                      'batch_size': self.batch_size,
                      'word_idx': self.word_idx,
                      'pretrained_emb_path': self.pretrained_emb_path,
                      'output_len': self.output_len,
                      'f_model': self.f_model,
                      'bidir': self.bidir,
                      'feature_idx': self.feature_idx,
                      'feat_size': self.feat_size,
                      'feat_padding_idx': self.feat_padding_idx,
                      'feat_emb_dim': self.feat_emb_dim
                      }

        # save model state
        state = {
            'net_params': net_params,
            'state_dict': self.state_dict(),
        }

        TorchUtils.save_model(state, f_model, dir_model)

    @classmethod
    def load(cls, f_model='lstm_pointer.tar', dir_model='../out/'):

        state = TorchUtils.load_model(f_model, dir_model)
        classifier = cls(**state['net_params'])
        classifier.load_state_dict(state['state_dict'])

        return classifier


class EncoderDecoder(nn.Module):
    def __init__(self,
                 n_layers,
                 hidden_dim,
                 vocab_size,
                 padding_idx,
                 label_padding_idx,
                 embedding_dim,
                 dropout,
                 batch_size,
                 word_idx,
                 label_idx,
                 pretrained_emb_path,
                 max_output_len,
                 label_size,
                 f_model,
                 bidir=False,
                 bert_embs=None,
                 constrained_decoding=None,
                 feature_idx=None,
                 feat_size=None,
                 feat_padding_idx=None,
                 feat_emb_dim=None,
                 feat_type=None,
                 feat_onehot=None,
                 cuda=0):
        """
        Initiate EncoderDecoder

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')
        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.label_padding_idx = label_padding_idx
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.word_idx = word_idx
        self.label_idx = label_idx
        self.pretrained_emb_path = pretrained_emb_path
        self.max_output_len = max_output_len
        self.n_labels = label_size
        self.f_model = f_model
        # decoder output length
        # if bidir:
        #    raise NotImplementedError
        self.bidir = bidir
        self.bert_embs = bert_embs
        self.constrained_decoding = constrained_decoding
        self.feature_idx = feature_idx
        self.feat_size = feat_size
        self.feat_padding_idx = feat_padding_idx
        self.feat_emb_dim = feat_emb_dim
        self.feat_type = feat_type
        self.feat_onehot = feat_onehot
        self.final_emb_dim = self.emb_dim + (
            self.feat_emb_dim * len(self.feat_type) if self.feat_emb_dim is not None else 0)

        self.encoder = Encoder(n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                               word_idx, pretrained_emb_path, bidir, bert_embs, feature_idx, feat_size,
                               feat_padding_idx,
                               feat_emb_dim, feat_type, feat_onehot, cuda=cuda)
        if self.constrained_decoding is not None:
            self.decoder = ConstrainedDecoderForSingleDec(hidden_dim, vocab_size, padding_idx, label_padding_idx,
                                                          embedding_dim, word_idx,
                                                          pretrained_emb_path,
                                                          max_output_len, label_size, label_idx, constrained_decoding,
                                                          cuda=cuda)
        else:
            self.decoder = Decoder(hidden_dim, vocab_size, padding_idx, label_padding_idx, embedding_dim, word_idx,
                                   pretrained_emb_path,
                                   max_output_len, label_size, cuda=cuda)
        self.decoder_input0 = Parameter(torch.FloatTensor(self.emb_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform(self.decoder_input0, -1, 1)
        self.to(self.device)

    #    def forward(self, inputs):
    def forward(self, sentence, features, sent_lengths, output_length=None, cur_labels=None, cur_max_nsymbs=None):
        # input_length = inputs.size(1)
        cur_batch_len = len(sent_lengths)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(cur_batch_len, -1)

        # inputs = inputs.view(batch_size * input_length, -1)
        # embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        enc_hidden0 = self.encoder.init_hidden()
        # encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, enc_hidden0)
        encoder_outputs, encoder_hidden = self.encoder(sentence, features, sent_lengths, enc_hidden0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        if self.bidir:
            decoder_hidden0 = (torch.cat(tuple(encoder_hidden[0][-2:]), dim=-1),
                               torch.cat(tuple(encoder_hidden[1][-2:]), dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        if cur_max_nsymbs is not None:
            (outputs, labels), decoder_hidden = self.decoder(sent_lengths, output_length,
                                                         decoder_input0,
                                                         decoder_hidden0,
                                                         encoder_outputs,
                                                         cur_labels,
                                                         cur_max_nsymbs)
        else:
            (outputs, labels), decoder_hidden = self.decoder(sent_lengths, output_length,
                                                             decoder_input0,
                                                             decoder_hidden0,
                                                             encoder_outputs,
                                                             cur_labels)

        return outputs, labels

    def loss(self, fwd_out, target):
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.label_padding_idx)
        return loss_fn(fwd_out, target)

    def train_model(self, corpus, dev_corpus, corpus_encoder, feature_encoder, n_epochs, optimizer):

        self.train()

        optimizer = optimizer
        best_acc = 0.

        for i in range(n_epochs):
            running_loss = 0.0

            # shuffle the corpus
            corpus.shuffle()
            # potential external features
            if feature_encoder is not None:
                _features = feature_encoder.get_feature_batches(corpus, self.batch_size, self.feat_type)
            else:
                _features = None
            if self.constrained_decoding is not None and "mod4" in self.constrained_decoding:
                # get constraint mod4:
                _max_nsymbs = get_max_nsymb_batches(corpus, self.batch_size)
            # get train batch
            for idx, (cur_insts, cur_labels) in enumerate(
                    corpus_encoder.get_batches(corpus, self.batch_size, token_ids=bool(self.bert_embs))):
                cur_feats = _features.__next__() if _features is not None else None
                if _features is not None:
                    assert len(cur_feats) == len(cur_insts)
                    cur_feats, cur_feat_lengths = feature_encoder.feature_batch_to_tensors(cur_feats, self.device,
                                                                                           len(self.feat_type))
                cur_max_nsymbs = _max_nsymbs.__next__() if (self.constrained_decoding is not None and "mod4" in self.constrained_decoding) else None
                cur_insts, cur_lengths, cur_labels, cur_label_lengths = corpus_encoder.batch_to_tensors(
                    cur_insts, cur_labels, self.device, padding_idx=0 if self.bert_embs else corpus_encoder.vocab.pad)
                output_length = max(cur_label_lengths).item()
                # forward pass
                fwd_out, labels = self.forward(cur_insts, cur_feats, cur_lengths,
                                               output_length=output_length,
                                               cur_labels=cur_labels,
                                               cur_max_nsymbs=cur_max_nsymbs)
                fwd_out = fwd_out.contiguous().view(-1, fwd_out.size()[-1])
                # loss calculation
                loss = self.loss(fwd_out, cur_labels.view(-1).long())

                # backprop
                optimizer.zero_grad()  # reset tensor gradients
                loss.backward()  # compute gradients for network params w.r.t loss
                optimizer.step()  # perform the gradient update step
                running_loss += loss.item()
            _y_pred, _y_true = self.predict(dev_corpus, feature_encoder, corpus_encoder)
            # for accuracy calculation
            y_true = [str(y) for y in _y_true]
            y_pred = [str(y) for y in _y_pred]
            self.train()  # set back the train mode
            dev_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            dev_f1 = np.mean([f1_score(y_true=t, y_pred=p) for t, p in zip(_y_true, _y_pred)])
            if i == 0 or dev_acc > best_acc:
                self.save(self.f_model)
                best_acc = dev_acc
            print('ep %d, loss: %.3f, dev_acc: %.3f, dev_f1: %.3f' % (i, running_loss, dev_acc, dev_f1))

    def predict(self, corpus, feature_encoder, corpus_encoder):
        self.eval()
        y_pred = list()
        y_true = list()

        # potential external features
        if feature_encoder is not None:
            _features = feature_encoder.get_feature_batches(corpus, self.batch_size, self.feat_type)
        else:
            _features = None
        # get constraint mod4:
        _max_nsymbs = get_max_nsymb_batches(corpus, self.batch_size)
        for idx, (cur_insts, cur_labels) in enumerate(
                corpus_encoder.get_batches(corpus, self.batch_size, token_ids=bool(self.bert_embs))):
            cur_feats = _features.__next__() if _features is not None else None
            if _features is not None:
                assert len(cur_feats) == len(cur_insts)
                cur_feats, cur_feat_lengths = feature_encoder.feature_batch_to_tensors(cur_feats, self.device,
                                                                                       len(self.feat_type))
            cur_max_nsymbs = _max_nsymbs.__next__() if (self.constrained_decoding is not None and "mod4" in self.constrained_decoding) else None
            cur_insts, cur_lengths, cur_labels, cur_label_lengths = corpus_encoder.batch_to_tensors(cur_insts,
                                                                                                    cur_labels,
                                                                                                    self.device,
                                                                                                    padding_idx=0 if self.bert_embs else corpus_encoder.vocab.pad)
            y_true.extend(cur_labels.cpu().numpy())

            # forward pass
            _, labels = self.forward(cur_insts, cur_feats, cur_lengths, cur_labels=cur_labels, cur_max_nsymbs=cur_max_nsymbs)

            y_pred.extend(labels.squeeze(1).cpu().numpy())
        y_true = corpus_encoder.strip_until_eos(y_true)
        y_pred = corpus_encoder.strip_until_eos(y_pred)
        return y_pred, y_true

    def save(self, f_model='lstm_encdec.tar', dir_model='../out/'):

        net_params = {'n_layers': self.n_lstm_layers,
                      'hidden_dim': self.hidden_dim,
                      'vocab_size': self.vocab_size,
                      'padding_idx': self.encoder.word_embeddings.padding_idx,
                      'label_padding_idx': self.label_padding_idx,
                      'embedding_dim': self.emb_dim,
                      'dropout': self.dropout,
                      'batch_size': self.batch_size,
                      'word_idx': self.word_idx,
                      'label_idx': self.label_idx,
                      'pretrained_emb_path': self.pretrained_emb_path,
                      'max_output_len': self.max_output_len,
                      'label_size': self.n_labels,
                      'f_model': self.f_model,
                      'bidir': self.bidir,
                      'bert_embs': self.bert_embs or None,
                      'constrained_decoding': self.constrained_decoding,
                      'feature_idx': self.feature_idx,
                      'feat_size': self.feat_size,
                      'feat_padding_idx': self.feat_padding_idx,
                      'feat_emb_dim': self.feat_emb_dim,
                      'feat_type': self.feat_type,
                      'feat_onehot': self.feat_onehot
                      }

        # save model state
        state = {
            'net_params': net_params,
            'state_dict': self.state_dict(),
        }

        TorchUtils.save_model(state, f_model, dir_model)

    @classmethod
    def load(cls, f_model='lstm_encdec.tar', dir_model='../out/'):

        state = TorchUtils.load_model(f_model, dir_model)
        classifier = cls(**state['net_params'])
        classifier.load_state_dict(state['state_dict'])

        return classifier

    @classmethod
    def remove(cls, f_model='lstm_encdec.tar'):
        os.remove("../out/" + f_model)


class EncoderSplitDecoder(nn.Module):
    def __init__(self,
                 n_layers,
                 hidden_dim,
                 vocab_size,
                 padding_idx,
                 label_padding_idx,
                 label_padding_idx2,
                 embedding_dim,
                 dropout,
                 batch_size,
                 word_idx,
                 label_idx,
                 label_idx2,
                 pretrained_emb_path,
                 max_output_len,
                 label_size,
                 label_size2,
                 f_model,
                 bidir=False,
                 bert_embs=None,
                 oracle_dec1=False,
                 constrained_decoding=None,
                 label_type_dec="full-pl-split",
                 cuda=0,
                 feature_idx=None,
                 feat_size=None,
                 feat_padding_idx=None,
                 feat_emb_dim=None,
                 feat_type=None,
                 feat_onehot=None):
        """
        Initiate EncoderDecoder

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """
        super().__init__()
        self.cuda = cuda
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.cuda}')
        else:
            self.device = torch.device('cpu')
        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.label_padding_idx = label_padding_idx
        self.label_padding_idx2 = label_padding_idx2
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.word_idx = word_idx
        self.label_idx = label_idx
        self.label_idx2 = label_idx2
        self.pretrained_emb_path = pretrained_emb_path
        self.max_output_len = max_output_len
        self.n_labels = label_size
        self.n_labels2 = label_size2
        self.f_model = f_model
        # decoder output length
        # if bidir:
        #    raise NotImplementedError
        self.bidir = bidir
        self.bert_embs = bert_embs
        self.oracle_dec1 = oracle_dec1
        self.constrained_decoding = constrained_decoding
        self.label_type_dec = label_type_dec
        self.feature_idx = feature_idx
        self.feat_size = feat_size
        self.feat_padding_idx = feat_padding_idx
        self.feat_emb_dim = feat_emb_dim
        self.feat_type = feat_type
        self.feat_onehot = feat_onehot
        self.final_emb_dim = self.emb_dim + (
            self.feat_emb_dim * len(self.feat_type) if self.feat_emb_dim is not None else 0)

        self.encoder = Encoder(n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                               word_idx, pretrained_emb_path, bidir, bert_embs, feature_idx, feat_size,
                               feat_padding_idx,
                               feat_emb_dim, feat_type, feat_onehot, cuda=cuda)
        self.decoder = Decoder(hidden_dim, vocab_size, padding_idx, label_padding_idx, embedding_dim, word_idx,
                               pretrained_emb_path,
                               max_output_len, label_size, cuda=cuda)
        # encoder for the decoded sequence, which is then passed to decoder 2
        self.encoder2 = Encoder(n_layers, hidden_dim, label_size, label_padding_idx, embedding_dim, dropout, batch_size,
                                word_idx, pretrained_emb_path=None, bidir=bidir, bert_embs=None, feature_idx=None,
                                feat_size=None,
                                feat_padding_idx=None, feat_emb_dim=None, feat_type=None, feat_onehot=None, cuda=cuda)
        if constrained_decoding is not None:
            self.decoder2 = ConstrainedDecoderForSplitDec(hidden_dim, vocab_size, padding_idx, label_padding_idx2,
                                                          embedding_dim, word_idx,
                                                          pretrained_emb_path, max_output_len, label_size2, label_size,
                                                          label_idx, label_idx2, constrained_decoding, cuda=cuda)
        else:
            self.decoder2 = Decoder(hidden_dim, vocab_size, padding_idx, label_padding_idx2, embedding_dim, word_idx,
                                    pretrained_emb_path, max_output_len, label_size2, cuda=cuda)

        self.decoder_input0 = Parameter(torch.FloatTensor(self.emb_dim), requires_grad=False)
        self.decoder_input02 = Parameter(torch.FloatTensor(self.emb_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform(self.decoder_input0, -1, 1)
        nn.init.uniform(self.decoder_input02, -1, 1)
        self.to(self.device)

    #    def forward(self, inputs):
    def forward(self, sentence, features, sent_lengths, output_length=None, output_length2=None, cur_labels=None,
                cur_labels2=None):
        # input_length = inputs.size(1)
        cur_batch_len = len(sent_lengths)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(cur_batch_len, -1)
        decoder_input02 = self.decoder_input02.unsqueeze(0).expand(cur_batch_len, -1)

        # inputs = inputs.view(batch_size * input_length, -1)
        # embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        # ENCODER
        enc_hidden0 = self.encoder.init_hidden()
        # encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, enc_hidden0)
        encoder_outputs, encoder_hidden = self.encoder(sentence, features, sent_lengths, enc_hidden0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # DECODER
        if self.bidir:
            decoder_hidden0 = (torch.cat(tuple(encoder_hidden[0][-2:]), dim=-1),  # final hidden state
                               torch.cat(tuple(encoder_hidden[1][-2:]), dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],  # final hidden state
                               encoder_hidden[1][-1])  # final cell state
        (outputs, labels), decoder_hidden = self.decoder(sent_lengths, output_length,
                                                         decoder_input0,
                                                         decoder_hidden0,
                                                         encoder_outputs,
                                                         cur_labels)

        # ENCODING THE DECODED SEQUENCE
        enc_hidden02 = self.encoder2.init_hidden()

        # obtain lengths of decoded label seqs, to use with encoder2
        if self.oracle_dec1:
            label_lengths = self.get_label_lengths(cur_labels)
            encoder_outputs2, encoder_hidden2 = self.encoder2(cur_labels.permute(1, 0), None, label_lengths,
                                                              enc_hidden02)

        else:
            label_lengths = self.get_label_lengths(labels)
            encoder_outputs2, encoder_hidden2 = self.encoder2(labels.permute(1, 0), None, label_lengths, enc_hidden02)
        encoder_outputs2 = encoder_outputs2.permute(1, 0, 2)

        # DECODER 2
        if self.bidir:
            decoder_hidden02 = (torch.cat(tuple(encoder_hidden2[0][-2:]), dim=-1),  # final hidden state
                                torch.cat(tuple(encoder_hidden2[1][-2:]), dim=-1))
        else:
            decoder_hidden02 = (encoder_hidden2[0][-1],  # final hidden state
                                encoder_hidden2[1][-1])  # final cell state
        label_lengths = self.get_label_lengths(labels)

        if self.constrained_decoding is not None:
            decoder_labels = labels
            decoder_outputs = outputs
            (outputs2, labels2), decoder_hidden2 = self.decoder2(label_lengths, output_length2, decoder_input02,
                                                                 decoder_hidden02,
                                                                 encoder_outputs2,
                                                                 cur_labels2,
                                                                 decoder_labels,
                                                                 self.decoder.label_embeddings,
                                                                 decoder_outputs)

        else:
            if self.label_type_dec == "full-pl-split-stat-dyn":
                dec_hidden0 = decoder_hidden0
                enc_outputs = encoder_outputs
            else:
                dec_hidden0 = decoder_hidden02
                enc_outputs = encoder_outputs2

            (outputs2, labels2), decoder_hidden2 = self.decoder2(label_lengths, output_length2, decoder_input02,
                                                                 dec_hidden0,
                                                                 enc_outputs,
                                                                 cur_labels2)

        return outputs, labels, outputs2, labels2

    def get_label_lengths(self, labels):
        """
        Obtain lengths of decoded label seqs, to use with encoder2
        """
        new_labels = []
        for ls in labels.cpu().numpy():
            new_ls = []
            for l in ls:
                if l == self.label_idx["</s>"]:
                    break
                new_ls.append(l)
            if not new_ls:
                new_ls.append(self.label_idx["<unk>"])
            new_labels.append(new_ls)
        label_lengths = [len(ls) for ls in new_labels]

        return torch.tensor(label_lengths, dtype=torch.int).to(self.device)

    def loss(self, fwd_out, target):
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.label_padding_idx)
        return loss_fn(fwd_out, target)

    def train_model(self, corpus, dev_corpus, corpus_encoder, feature_encoder, n_epochs, optimizer):

        self.train()

        optimizer = optimizer
        best_acc = 0.

        for i in range(n_epochs):
            running_loss = 0.0

            # shuffle the corpus
            corpus.shuffle()
            # potential external features
            if feature_encoder is not None:
                _features = feature_encoder.get_feature_batches(corpus, self.batch_size, self.feat_type)
            else:
                _features = None
            # get train batch
            for idx, (cur_insts, cur_labels, cur_labels2) in enumerate(
                    corpus_encoder.get_batches(corpus, self.batch_size, token_ids=bool(self.bert_embs))):
                cur_feats = _features.__next__() if _features is not None else None
                if _features is not None:
                    assert len(cur_feats) == len(cur_insts)
                    cur_feats, cur_feat_lengths = feature_encoder.feature_batch_to_tensors(cur_feats, self.device,
                                                                                           len(self.feat_type))
                cur_insts, cur_lengths, cur_labels, cur_label_lengths, cur_labels2, cur_label_lengths2 = corpus_encoder.batch_to_tensors(
                    cur_insts, cur_labels, cur_labels2, self.device,
                    padding_idx=0 if self.bert_embs else corpus_encoder.vocab.pad)
                output_length = max(cur_label_lengths).item()
                output_length2 = max(cur_label_lengths2).item()
                if self.oracle_dec1:
                    assert output_length == output_length2
                # forward pass
                fwd_out, labels, fwd_out2, labels2 = self.forward(cur_insts, cur_feats, cur_lengths,
                                                                  output_length=output_length,
                                                                  output_length2=output_length2,
                                                                  cur_labels=cur_labels, cur_labels2=cur_labels2)
                fwd_out = fwd_out.contiguous().view(-1, fwd_out.size()[-1])
                fwd_out2 = fwd_out2.contiguous().view(-1, fwd_out2.size()[-1])
                # loss calculation
                loss = self.loss(fwd_out, cur_labels.view(-1).long())
                loss2 = self.loss(fwd_out2, cur_labels2.view(-1).long())

                # print(f"loss1: {loss}; loss2: {loss2}")
                total_loss = loss + loss2

                # backprop
                optimizer.zero_grad()  # reset tensor gradients
                total_loss.backward()  # compute gradients for network params w.r.t loss
                optimizer.step()  # perform the gradient update step
                running_loss += total_loss.item()
            _y_pred, _y_true, (_y_pred1, _y_pred2, _y_true1, _y_true2) = self.predict(dev_corpus, feature_encoder,
                                                                                      corpus_encoder)
            # for accuracy calculation
            y_true = [str(y) for y in _y_true]
            y_pred = [str(y) for y in _y_pred]
            y_pred1 = [str(y) for y in _y_pred1]
            y_pred2 = [str(y) for y in _y_pred2]
            y_true1 = [str(y) for y in _y_true1]
            y_true2 = [str(y) for y in _y_true2]
            self.train()  # set back the train mode
            dev_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            dev_acc1 = accuracy_score(y_true=y_true1, y_pred=y_pred1)
            dev_acc2 = accuracy_score(y_true=y_true2, y_pred=y_pred2)
            # print(f"acc dec1: {dev_acc1}, acc dec2: {dev_acc2}")
            dev_f1 = np.mean([f1_score(y_true=t, y_pred=p) for t, p in zip(_y_true, _y_pred)])
            if i == 0 or dev_acc > best_acc:
                self.save(self.f_model)
            print('ep %d, loss: %.3f, dev_acc1: %.3f, dev_acc2: %.3f. dev_acc: %.3f, dev_f1: %.3f' % (
            i, running_loss, dev_acc1, dev_acc2, dev_acc, dev_f1))

    def predict(self, corpus, feature_encoder, corpus_encoder):
        def join_dec_labels(l1, l2, exclude_type_str=False, compact=False):
            """
            :param l1: label list from decoder1
            :param l2: label list from decoder2
            :param exclude_type_str: when pasting don't include type str from dec1 as dec2 already has it
            :return: joined list of labels

            Join labels from decoder1 and decoder2:
            - substitute COPY labels in decoder2 with those from decoder1
            - merge int labels from decoder2 with l/n labels from decoder1
            """
            l1_map = corpus_encoder.label_vocab.idx2word
            l2_map = corpus_encoder.label_vocab2.idx2word
            # if exclude_type_str:
            #    ln_idxs = set()
            #    for k,v in corpus_encoder.label_vocab.word2idx.items():
            #        h = re.findall("^l|n\d+$", k)
            #        if h:
            #            ln_idxs.add(h.pop())
            # else:
            #    l_idx = corpus_encoder.label_vocab.word2idx["l"]
            #    n_idx = corpus_encoder.label_vocab.word2idx["n"]
            #    ln_idxs = {l_idx, n_idx}
            l_idx = corpus_encoder.label_vocab.word2idx["l"]
            n_idx = corpus_encoder.label_vocab.word2idx["n"]
            ln_idxs = {l_idx, n_idx}

            l = []
            for m, x1 in enumerate(l1):
                r = []
                ix1_cnt = 0
                for n, ix1 in enumerate(x1):
                    if ix1 in ln_idxs:  # need a number here
                        ix1_cnt += 1
                        ix1_m = "" if exclude_type_str else l1_map[ix1]
                        try:
                            l2_str = l2_map[l2[m, ix1_cnt]] if compact else l2_map[l2[m, n]]
                            r.append(ix1_m + l2_str)
                        except IndexError:
                            r.append(l1_map[ix1] + "0")
                    else:
                        r.append(l1_map[ix1])
                l.append(r)
            return l

        def join_stat_dyn_labels(l1, l2):
            """
            :param l1: label list from decoder1
            :param l2: label list from decoder2
            :return: joined list of labels

            Join labels from decoder1 and decoder2:
            -
            -
            """
            l1_map = corpus_encoder.label_vocab.idx2word
            l2_map = corpus_encoder.label_vocab2.idx2word
            l = []
            for _l1, _l2 in zip(l1, l2):
                l.append([l1_map[i] for i in _l1] + [l2_map[i] for i in _l2])
            return l

        self.eval()
        y_pred = list()
        y_true = list()

        # potential external features
        if feature_encoder is not None:
            _features = feature_encoder.get_feature_batches(corpus, self.batch_size, self.feat_type)
        else:
            _features = None
        for idx, (cur_insts, cur_labels, cur_labels2) in enumerate(
                corpus_encoder.get_batches(corpus, self.batch_size, token_ids=bool(self.bert_embs))):
            cur_feats = _features.__next__() if _features is not None else None
            if _features is not None:
                assert len(cur_feats) == len(cur_insts)
                cur_feats, cur_feat_lengths = feature_encoder.feature_batch_to_tensors(cur_feats, self.device,
                                                                                       len(self.feat_type))
            cur_insts, cur_lengths, cur_labels, cur_label_lengths, cur_labels2, cur_label_lengths2 = corpus_encoder.batch_to_tensors(
                cur_insts,
                cur_labels,
                cur_labels2,
                self.device,
                padding_idx=0 if self.bert_embs else corpus_encoder.vocab.pad)

            # forward pass
            _, labels, _, labels2 = self.forward(cur_insts, cur_feats, cur_lengths, cur_labels=cur_labels,
                                                 cur_labels2=cur_labels2)

            if self.label_type_dec == "full-pl-split":
                y_true.extend(join_dec_labels(corpus_encoder.strip_until_eos(cur_labels.cpu().numpy()),
                                              cur_labels2.cpu().numpy()))
            elif self.label_type_dec == "full-pl-split-plc":
                y_true.extend(join_dec_labels(corpus_encoder.strip_until_eos(cur_labels.cpu().numpy()),
                                              cur_labels2.cpu().numpy(), exclude_type_str=True, compact=True))
            elif self.label_type_dec == "full-pl-split-stat-dyn":
                y_true.extend(join_stat_dyn_labels(corpus_encoder.strip_until_eos(cur_labels.cpu().numpy()),
                                                   corpus_encoder.strip_until_eos(cur_labels2.cpu().numpy())))
            else:
                raise ValueError

            if self.oracle_dec1:
                if self.label_type_dec == "full-pl-split-stat-dyn":
                    raise NotImplementedError
                y_pred.extend(join_dec_labels(corpus_encoder.strip_until_eos(cur_labels.squeeze(1).cpu().numpy()),
                                              labels2.squeeze(1).cpu().numpy()))
            else:
                if self.label_type_dec == "full-pl-split":
                    y_pred.extend(join_dec_labels(corpus_encoder.strip_until_eos(labels.squeeze(1).cpu().numpy()),
                                                  labels2.squeeze(1).cpu().numpy()))
                elif self.label_type_dec == "full-pl-split-plc":
                    y_pred.extend(join_dec_labels(corpus_encoder.strip_until_eos(labels.squeeze(1).cpu().numpy()),
                                                  labels2.squeeze(1).cpu().numpy(), exclude_type_str=True,
                                                  compact=True))
                elif self.label_type_dec == "full-pl-split-stat-dyn":
                    y_pred.extend(join_stat_dyn_labels(corpus_encoder.strip_until_eos(labels.squeeze(1).cpu().numpy()),
                                                       corpus_encoder.strip_until_eos(
                                                           labels2.squeeze(1).cpu().numpy())))
                else:
                    raise ValueError
        return y_pred, y_true, (
        corpus_encoder.strip_until_eos(labels.cpu().numpy()), corpus_encoder.strip_until_eos(labels2.cpu().numpy()),
        corpus_encoder.strip_until_eos(cur_labels.cpu().numpy()),
        corpus_encoder.strip_until_eos(cur_labels2.cpu().numpy()))

    def save(self, f_model='lstm_encsplitdec.tar', dir_model='../out/'):

        net_params = {'n_layers': self.n_lstm_layers,
                      'hidden_dim': self.hidden_dim,
                      'vocab_size': self.vocab_size,
                      'padding_idx': self.encoder.word_embeddings.padding_idx,
                      'label_padding_idx': self.label_padding_idx,
                      'label_padding_idx2': self.label_padding_idx2,
                      'embedding_dim': self.emb_dim,
                      'dropout': self.dropout,
                      'batch_size': self.batch_size,
                      'word_idx': self.word_idx,
                      'label_idx': self.label_idx,
                      'label_idx2': self.label_idx2,
                      'pretrained_emb_path': self.pretrained_emb_path,
                      'max_output_len': self.max_output_len,
                      'label_size': self.n_labels,
                      'label_size2': self.n_labels2,
                      'f_model': self.f_model,
                      'bidir': self.bidir,
                      'bert_embs': self.bert_embs or None,
                      'oracle_dec1': self.oracle_dec1,
                      'constrained_decoding': self.constrained_decoding,
                      'label_type_dec': self.label_type_dec,
                      'cuda': self.cuda,
                      'feature_idx': self.feature_idx,
                      'feat_size': self.feat_size,
                      'feat_padding_idx': self.feat_padding_idx,
                      'feat_emb_dim': self.feat_emb_dim,
                      'feat_type': self.feat_type,
                      'feat_onehot': self.feat_onehot
                      }

        # save model state
        state = {
            'net_params': net_params,
            'state_dict': self.state_dict(),
        }

        TorchUtils.save_model(state, f_model, dir_model)

    @classmethod
    def load(cls, f_model='lstm_encsplitdec.tar', dir_model='../out/'):

        state = TorchUtils.load_model(f_model, dir_model)
        classifier = cls(**state['net_params'])
        classifier.load_state_dict(state['state_dict'])

        return classifier

    @classmethod
    def remove(cls, f_model='lstm_encsplitdec.tar'):
        os.remove("../out/" + f_model)
