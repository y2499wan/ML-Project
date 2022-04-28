# These are the hyperparameters used in test.py
stock_name = 'PFE'

window_size = 128
enc_seq_len = window_size
dec_seq_len = 2
output_sequence_length = 1

input_size = 5
dim_val = 64 # embedding size
dim_attn = 128
lr = 0.001
epochs = 20

n_heads = 8

n_decoder_layers = 2
n_encoder_layers = 3
batch_size = 64


# time to vec
time_embed_size = 2