""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class word_memory(nn.Module):
    def __init__(self, embedding_a, embedding_b, embedding_c):
        super(word_memory, self).__init__()
        # self.config = config
        # self.args = args
        self.word_embedding_a = embedding_a
        self.word_embedding_b = embedding_b
        self.word_embedding_c = embedding_c
        # self.linear_1 = nn.Linear(config.word_embedding_dim, config.hidden_size, bias=False)
        # self.linear_2 = nn.Linear(config.word_embedding_dim, 64)

    def forward(self, word_seq, all_docs):
        '''
            word_seq: [300,13,1] [seq_len, batch_size, 1]
            all_docs: [300,1100,1] [seq_len, all_docs_num, 1]
        '''
        # ci = torch.mean(self.word_embedding_a(all_docs), 0) #[300,1100,512]
        # mi = torch.mean(self.word_embedding_c(all_docs), 0) #[300,1100,512]


       # @shizhe.  success by slicing

        slice_num = 40
        u = torch.mean(self.word_embedding_b(word_seq), 0) #[13,512]       [300,13,512] [seq_len, batch_size, emb_dim] -> [13,512]
        all_docs = all_docs.permute(1, 0, 2)   #[1100,300,1]
        c = torch.mean(self.word_embedding_a(all_docs[0*slice_num:1*slice_num]), 1) #[4000, 512]  [1100,300,1] - > [1100,300,512] -> [1100,512]
        m = torch.mean(self.word_embedding_a(all_docs[0*slice_num:1*slice_num]), 1) #[4000, 512]  [1100,300,1] - > [1100,300,512] -> [1100,512]
        num_loop = math.ceil(all_docs.shape[0] / slice_num)
        for i in range(1, num_loop):
            #print("i: ", i)
            doci = all_docs[i * slice_num:(i + 1) * slice_num]
            ci = self.word_embedding_a(doci) #all_docs[1100,300,1] ->[slice_num,300,1] -> emb = [slice_num,300,512]
            mi = self.word_embedding_c(doci) #all_docs[1100,300,1] ->[slice_num,300,1] -> emb = [slice_num,300,512]
            ci = torch.mean(ci, 1)  #[slice_num,512]
            mi = torch.mean(mi, 1)  #[slice_num,512]
            c = torch.cat([c, ci])
            m = torch.cat([m, mi])
            del ci,mi,doci
            #torch.cuda.empty_cache()

        # for i in range(1, all_docs.shape[0]):
        #     ci = torch.add(ci, self.word_embedding_a(all_docs[i].unsqueeze(2)))
        #     mi = torch.add(mi, self.word_embedding_c(all_docs[i].unsqueeze(2)))
        # ci = (ci / all_docs.shape[0]).squeeze(1)
        # mi = (mi / all_docs.shape[0]).squeeze(1)

        #get sentence embedding by average
        # u = torch.mean(u,0) #[13,512]
        # c = torch.mean(c,0) #[1100,512]
        # m = torch.mean(m,0) #[1100,512]

        m = m.permute(1,0) #[1100, 512] -  [512,1100]
        p = F.softmax(torch.matmul(u, m), dim=1) #[13,1100]


        # print('attention shape', attention.shape)
        # p (batch_size, character_seq_len, word_seq_len, hidden_state)
        #p = torch.stack([p] * ci.shape[1], 2) #[13,1100, 512]

        # character_attention (batch_size, character_seq_len, word_seq_len, word_embedding_dim)
        o = torch.matmul(p, c) #[13,512]
        # print('character attention shape', character_attention.shape)
        #o = torch.add(o, hidden_state)
        del c, m, p, u, all_docs
        return o

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, all_docs=None, model_opt=None, embedding_a=None, embedding_b=None, embedding_c=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.all_docs = all_docs
        #self.cal_word_memory = word_memory(embedding_a, embedding_b, embedding_c)

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        # src [300,13,1] [seq_len, batch_size, 1]
        # tgt [5,13,1] [tgt_len, batch, 1]
        #enc_state is the encoding of src [300,13,512]  memory_bank is the hidden state of last layer [300,13,512] lengths [13]
        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        '''
        @shizhe failed try
        docs = self.all_docs
        docs_lengths = torch.ones(docs.shape[1])*300
        docs_lengths = docs_lengths.to(lengths.device)
        enc_state, memory_bank, lengths = self.encoder(docs, docs_lengths)
        '''
        #******************@shizhe**************
        # if self.all_docs is not None:
        #     o = self.cal_word_memory(src, self.all_docs)  #[13,512]  [batch_size, emb_dim]
        #     memory_bank = torch.add(memory_bank, o)
        #******************@shizhe**************
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
