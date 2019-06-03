import random

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

from util import TorchUtils, load_emb


class Encoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                 word_idx, pretrained_emb_path, feature_idx, feat_size, feat_padding_idx, feat_emb_dim):
        super().__init__()
        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.word_idx = word_idx
        self.pretrained_emb_path = pretrained_emb_path
        self.feature_idx = feature_idx
        self.feat_size = feat_size
        self.feat_padding_idx = feat_padding_idx
        self.feat_emb_dim = feat_emb_dim
        self.final_emb_dim = self.emb_dim + (self.feat_emb_dim if self.feat_emb_dim is not None else 0)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.hidden_in = self.init_hidden()  # initialize cell states

        if pretrained_emb_path is not None:
            self.word_embeddings, dim = load_emb(pretrained_emb_path, word_idx, freeze=False)
            assert dim == self.emb_dim
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim,
                                                padding_idx=padding_idx)  # embedding layer, initialized at random

        if feature_idx is not None:
            self.feat_embeddings = nn.Embedding(self.feat_size, self.feat_emb_dim, padding_idx=feat_padding_idx)

        self.lstm = nn.LSTM(self.final_emb_dim, self.hidden_dim, num_layers=self.n_lstm_layers,
                            dropout=self.dropout)  # lstm layers
        self.to(self.device)

    def init_hidden(self):
        '''
        initializes hidden and cell states to zero for the first input
        '''
        h0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).to(self.device)

        return (h0, c0)

    def forward(self, sentence, features, sent_lengths, hidden):
        sort, unsort = TorchUtils.get_sort_unsort(sent_lengths)
        embs = self.word_embeddings(sentence).to(self.device)  # word sequence to embedding sequence
        if features is not None:
            feat_embs = self.feat_embeddings(features).to(self.device)  # feature sequence to embedding sequence
            embs = torch.cat([embs, feat_embs], dim=2).to(self.device)

        # truncating the batch length if last batch has fewer elements
        cur_batch_len = len(sent_lengths)
        hidden = (hidden[0][:, :cur_batch_len, :], hidden[1][:, :cur_batch_len, :])

        # converts data to packed sequences with data and batch size at every time step after sorting them per lengths
        embs = nn.utils.rnn.pack_padded_sequence(embs[:, sort], sent_lengths[sort], batch_first=False)

        # lstm_out: output of last lstm layer after every time step
        # hidden gets updated and cell states at the end of the sequence
        lstm_out, hidden = self.lstm(embs, hidden)
        # pad the sequences again to convert to original padded data shape
        lstm_out, lengths = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=False)

        # unsort batch
        lstm_out = lstm_out[:, unsort]
        hidden = (hidden[0][:, unsort, :], hidden[1][:, unsort, :])

        return lstm_out, hidden


class LSTMClassifier(nn.Module):
    # based on https://github.com/MadhumitaSushil/sepsis/blob/master/src/classifiers/lstm.py
    def __init__(self, n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, label_size, batch_size,
                 word_idx, pretrained_emb_path):
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

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.encoder = Encoder(n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                               word_idx, pretrained_emb_path)
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
            if dev_acc > best_acc:
                self.save()
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
                      'pretrained_emb_path': self.pretrained_emb_path
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
                 pretrained_emb_path):
        super().__init__()

        self.n_lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.word_idx = word_idx
        self.pretrained_emb_path = pretrained_emb_path

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if pretrained_emb_path is not None:
            self.word_embeddings, dim = load_emb(pretrained_emb_path, word_idx, freeze=False)
            assert dim == self.emb_dim
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim,
                                                padding_idx=padding_idx)  # embedding layer, initialized at random

        self.encoder = Encoder(n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                               word_idx, pretrained_emb_path)

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
            if dev_mse < best_mse:
                self.save()
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
                      'pretrained_emb_path': self.pretrained_emb_path
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
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
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
                mask):
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
        if len(att[mask]) > 0:
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
                 feature_idx, feat_size, feat_padding_idx, feat_emb_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
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
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.final_emb_dim).byte()
            decoder_input = embs[embedding_mask.data].view(cur_batch_len, self.final_emb_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)  # (b * output_len * input_len)
        pointers = torch.cat(pointers, 1)  # (b * output_len)

        return (outputs, pointers), hidden


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
                 bidir=False,
                 feature_idx=None,
                 feat_size=None,
                 feat_padding_idx=None,
                 feat_emb_dim=None):
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
            self.device = torch.device('cuda:0')
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
                               feat_emb_dim)
        self.decoder = PointerDecoder(hidden_dim, vocab_size, padding_idx, embedding_dim, word_idx, pretrained_emb_path,
                                      output_len, feature_idx, feat_size, feat_padding_idx, feat_emb_dim)
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
            self.train()  # set back the train mode
            dev_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            if dev_acc > best_acc:
                self.save()
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

        if self.output_len > 1:
            y_true = [str(y) for y in y_true]
            y_pred = [str(y) for y in y_pred]

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
    def load(cls, f_model='lstm_classifier.tar', dir_model='../out/'):

        state = TorchUtils.load_model(f_model, dir_model)
        classifier = cls(**state['net_params'])
        classifier.load_state_dict(state['state_dict'])

        return classifier
