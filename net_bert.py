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


# pip install transformers
from transformers import BertConfig, BertModel, BertTokenizer


"""
# example of BERT use:
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentences = ['This is sentence 1.', 'This is sentence 2']
encoding = tokenizer(sentences,
                     max_length=500,
                     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                     return_token_type_ids=False,
                     pad_to_max_length=True,
                     return_attention_mask=True,
                     return_tensors='pt',  # Return PyTorch tensors
                     truncation=True
                     )
last_hidden_state, pooled_output = self.encoder(input_ids=encoding['input_ids'],
                                                 attention_mask=encoding['attention_mask'])                                          
"""


class BertEncoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, padding_idx, embedding_dim, dropout, batch_size,
                 word_idx, pretrained_emb_path, bidir, bert_embs, feature_idx, feat_size, feat_padding_idx,
                 feat_emb_dim, feat_type, feat_onehot, cuda=0, **kwargs):
        super().__init__()

        self.hidden_dim = 768
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.bidir = False

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

        if feature_idx is not None:
            if self.feat_onehot:
                self.feat_embeddings = self.one_hot
            else:
                self.feat_embeddings = nn.Embedding(self.feat_size, self.feat_emb_dim, padding_idx=feat_padding_idx)

        # this loads the pretrained BERT model from huggingface transformers library
        self.encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def embedded_dropout(self, embed, words, dropout=0.1, scale=None):
        pass

    def init_hidden(self):
        pass

    def one_hot(self, features):
        pass

    def forward(self, sentence, features, sent_lengths, hidden, training):

        # sentence = LongTensor of word embedding idxs, shape (num_samples, num_idxs)
        # REQUIRED: convert sentence back to list of sentences to encode
        # e.g. ['This is sentence 1.', 'This is sentence 2']
        sentences = []

        # tokenize the list of sentences
        encoding = self.bert_tokenizer(sentences,
                                     max_length=500,
                                     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                     return_token_type_ids=False,
                                     pad_to_max_length=True,
                                     return_attention_mask=True,
                                     return_tensors='pt',  # Return PyTorch tensors
                                     truncation=True
                                     ).to(self.device)

        # forward pass
        # pooled output is by default the encoding of the first token, [CLS]
        # it is performed by BertPooler :
        # https://github.com/huggingface/transformers/blob/edf0582c0be87b60f94f41c659ea779876efc7be/src/transformers/modeling_bert.py#L426

        last_hidden_states, pooled_output = self.encoder(input_ids=encoding['input_ids'],
                                                 attention_mask=encoding['attention_mask'])
        bert_out = pooled_output
        hidden = last_hidden_states

        return bert_out, hidden
