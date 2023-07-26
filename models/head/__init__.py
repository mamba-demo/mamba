import torch.nn as nn
from .AttnMIL import AttnMeanPoolMIL, SumMIL, HierarchicalAttnMeanPoolMIL, AdditiveMIL

def get_decoder_input_dim(encoder='resnet18_3d'):
    if 'resnet18' in encoder:
        input_dim = 512
    elif 'resnet34' in encoder:
        input_dim = 512
    elif 'resnet50' in encoder:
        input_dim = 1024
    elif encoder == 'SwinUNETR':
        input_dim = 768
    elif encoder == '2plus1d':
        input_dim = 1024
    elif encoder == 'clip':
        input_dim = 768
    elif encoder == 'slowfast':
        input_dim = 1152
    elif encoder == 'swin3d_s' or encoder == 'swin2d_s':
        input_dim = 768
    elif encoder == 'swin3d_b' or encoder == 'swin2d_b':
        input_dim = 1024
    else:
        raise NotImplementedError("Not implemented!")

    return input_dim


def get_decoder_model(decoder='attn',
                      input_dim=128,
                      out_dim=2,
                      attn_latent_dim=32,
                      dropout=0.25,
                      warm_start=False,
                      decoder_enc=False,
                      decoder_enc_num=1,
                      decoder_enc_dim=128,
                      context=False,
                      context_network='GRU',
                      **kwargs):
    """
    Get decoder: Features -> outcome
    """

    # Small encoder (1-layer MLP within the decoder)
    if decoder_enc:
        if decoder_enc_num == 0:
            decoder_enc = None
        elif decoder_enc_num == 1: # simplify
            decoder_enc = nn.Sequential(
                                        nn.Linear(input_dim, decoder_enc_dim),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )
        elif decoder_enc_num == 2: # simplify
            decoder_enc = nn.Sequential(
                                        nn.Linear(input_dim, decoder_enc_dim),
                                        nn.GELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(decoder_enc_dim, decoder_enc_dim),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )
        else:
            raise NotImplementedError("not implemented!")

    else:
        decoder_enc = None

    if decoder == 'attn':
        model = AttnMeanPoolMIL(gated=True,
                                encoder_dim=decoder_enc_dim if decoder_enc else input_dim,
                                encoder=decoder_enc,
                                attn_latent_dim=attn_latent_dim,
                                dropout=dropout,
                                warm_start=warm_start,
                                out_dim=out_dim)

    elif decoder == 'additive':
        model = AdditiveMIL(gated=True,
                            encoder_dim=decoder_enc_dim if decoder_enc else input_dim,
                            encoder=decoder_enc,
                            attn_latent_dim=attn_latent_dim,
                            dropout=dropout,
                            warm_start=warm_start,
                            out_dim=out_dim)

    elif 'hierarchical' in decoder:
        inter_mode = decoder.split('_')[1]
        assert inter_mode in ['avg', 'max', 'attn', 'gated'], "Attn inter mode has to be one of four options"

        model = HierarchicalAttnMeanPoolMIL(gated=True,
                                            encoder_dim=decoder_enc_dim if decoder_enc else input_dim,
                                            encoder=decoder_enc,
                                            attn_latent_dim=attn_latent_dim,
                                            attn_inter_mode=inter_mode,
                                            dropout=dropout,
                                            out_dim=out_dim,
                                            warm_start=warm_start,
                                            context=context,
                                            context_network=context_network)

    elif decoder == 'uniform':
        # Uniform attention weights
        model = SumMIL(encoder_dim=decoder_enc_dim if decoder_enc else input_dim,
                       out_dim=out_dim,
                       encoder=decoder_enc)

    else:
        raise NotImplementedError("{} not implemented!".format(decoder))

    return model
