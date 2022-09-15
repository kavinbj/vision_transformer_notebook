'''
Author: kavinbj
Date: 2022-09-08 17:26:24
LastEditTime: 2022-09-11 15:05:34
FilePath: __init__.py
Description: 

Copyright (c) 2022 by kavinbj, All Rights Reserved. 
'''

from torch import nn
from transformer.model.encoder import TransformerEncoder
from transformer.model.decoder import TransformerDecoder
from transformer.model.masked_softmax import sequence_mask

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens, *args):
        enc_outputs = self.encoder(enc_X, enc_valid_lens, *args)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens, *args)
        return self.decoder(dec_X, dec_state)


__all__ = [
    'TransformerEncoder',
    'TransformerDecoder',
    'EncoderDecoder',
    'sequence_mask'
]