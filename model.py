import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#need to fix the emnedding stuff (make it take glove weights)
class attention(nn.module):
	def __init__(self, opts):
		super(attention, self).__init__()
		self.Uz = torch.randn(opts.coattention_ninp, opts.final_dim)
		self.Ug = torch.randn(opts.coattention_ninp, opts.final_dim)
		self.ba = torch.randn(opts.coattention_ninp, 1)
		self.ua = torch.randn(opts.coattention_ninp, 1)

	

	def forward(self, Z, g):
		out1 = torch.matmul(self.Uz, Z)
		out2 = torch.matmul(self.Ug, g)
		out = torch.tanh(out1+out2+self.ba)
		az = torch.softmax(torch.matmul(self.ua.T, out),2)
		return az, torch.bmm(Z,az.permute(0,2,1)).squeeze()

#https://www.youtube.com/watch?v=qTn1Akspipg

class ABLR(nn.module):
	def __init__(self, opts):
		super(ABLR, self).__init__()

		## caption encoding
		self.vocab_size = opts.vocab_size
		self.emb_dropout = nn.Dropout(opts.rnn_dropout)
		self.cap_enc = nn.Embedding(opts.vocab_size, opts.cap_ninp, padding_idx = 0)
		self.cap_enc.load_state_dict({'weight':weights_matrix})
		self.cap_rnn = getattr(nn, opts.rnn_type)(opts.cap_ninp, opts.cap_nhid, opts.rnn_nlayers, \
											dropout=opts.rnn_dropout, batch_first=True, bidirectional=True)

		self.cap_process = nn.Linear(2*opts.cap_nhid, opts.final_dim)


		## video encoding
		self.vid_rnn = getattr(nn, opts.rnn_type)(opts.vid_ninp, opts.vid_nhid, opts.rnn_nlayers, \
											dropout=opts.rnn_dropout, batch_first=True, bidirectional=True)

		self.vid_process = nn.Linear(2*opts.vid_nhid, opts.final_dim)

		## co-attention
		self.coattention = Coattention(opts)
		

		## prediction layers
		self.pred_dropout = nn.Dropout(opts.pred_dropout)
		self.fc1 = nn.Linear(opts.pred_ninp, opts.pred_nhid)
		self.fc2 = nn.Linear(opts.pred_nhid, 2)

		def init_weights(m):
			#m.weight.data.uniform_(-0.1, 0.1)
			nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            
        self.apply(init_weights)


	def forward(self, opts, vid, cap, mask):
		"""
		vid and cap should be of dim (batch_size, seq_len, input_size)
		"""
		cap = self.emb_dropout(self.cap_enc(cap))
		cap_hidden = torch.zeros(opts.rnn_layers, opts.batch_size, opts.cap_nhid)
		cap, _ = self.cap_rnn(cap, cap_hidden)
		cap_dim = cap.size()
		cap = self.cap_process(cap.contiguous().view(-1, 2*opts.cap_nhid))
		cap = cap.view(cap_dim)
		cap = mask*cap.permute(2,0,1)
		cap = cap.permute(1,2,0)

		vid_hidden = torch.zeros(opts.rnn_layers, opts.batch_size, opts.vid_nhid)
		vid, _ = self.vid_rnn(vid, vid_hidden)
		vid_dim = vid.size()
		vid = self.vid_process(vid.contiguous().view(-1, 2*self.opts.vid_nhid))
		vid = vid.view(cap_dim)


		cap_mean = cap.mean(dim=1)
		vid_temp = vid.permute(0,2,1)
		cap_temp = cap.permute(0,2,1)

		attention = attention(opts)
		_, vid_att = attention.forward(opts, vid_temp, cap_mean)
		_, cap_att = attention.forward(opts, cap_temp, vid_att)
		att_weights, vid_att = attention.forward(opts, vid_temp, cap_att)

		out = self.pred_dropout(att_weights)
		out = nn.functional.relu(self.fc1(out))
		out = self.pred_dropout(out)
		out = nn.functional.sigmoid(self.fc2(out))

		return att_weights, out


