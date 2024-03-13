import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super(PositionalEmbedding, self).__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)
        self.num_vocab = num_vocab
        self.maxlen = maxlen
        self.num_hid = num_hid

    def forward(self, x):
        maxlen = x.size(-1)
        x = self.emb(x)
        positions = torch.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        return x + positions


class FixedEmbedding(nn.Module):
    def __init__(self, maxlen=100, num_hid=64, device="cpu"):
        super(FixedEmbedding, self).__init__()
        self.maxlen = maxlen
        self.num_hid = num_hid
        self.device=device
        self.pos_encoding = self.positional_encoding(self.maxlen, self.num_hid)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / float(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(torch.arange(position,device=self.device).unsqueeze(1), torch.arange(d_model,device=self.device).unsqueeze(0), d_model)
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads.unsqueeze(0)
        return pos_encoding#.float()

    def forward(self, x):
        bs, seq_len = x.size(0), x.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :].repeat(bs, 1, 1)
        x = torch.cat((x, pos_encoding.to(self.device)), dim=-1)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        attention_output, _ = self.attention(inputs, inputs, inputs, key_padding_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
        

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, seq_length, device):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.device = device
        self.attention_1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.attention_2 = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.layernorm_3 = nn.LayerNorm(embed_dim)
        
        self.register_buffer("causal_mask", self.generate_causal_mask(seq_length))

    def generate_causal_mask(self, seq_len):
        # Generate a causal mask to mask out future positions
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),diagonal=1)
        return mask
        
    def forward(self, inputs, encoder_outputs, mask=None):
        # Use the precomputed causal mask here
        causal_mask = self.causal_mask[:inputs.size(1), :inputs.size(1)]
        
        if mask is not None:
            padding_mask = mask.unsqueeze(1)
            padding_mask = torch.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None
            
        inputs=inputs.permute(1,0,2) #permute because first attention layer, due to attention mask, cannot be used with argument batch_first=True
        attention_output_1, _ = self.attention_1(inputs, inputs, inputs, attn_mask=causal_mask)
        attention_output_1=attention_output_1.permute(1,0,2) #permute back to (N,S,E)
        inputs=inputs.permute(1,0,2) #permute back
        
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2, _ = self.attention_2(attention_output_1, encoder_outputs, encoder_outputs, key_padding_mask=padding_mask)
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


class TransformerModel(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, seq_length, device):
        super(TransformerModel, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.encoder = TransformerEncoder(embed_dim*2, dense_dim, num_heads)
        self.decoder = TransformerDecoder(embed_dim*2, dense_dim, num_heads, seq_length,device=device)
        self.conv1d = nn.Conv1d(embed_dim * 2, 2, kernel_size=1, padding='same')
        self.embedding1=FixedEmbedding(maxlen=(9900-5000)//2, num_hid=embed_dim-1,device = device)
        self.embedding2=FixedEmbedding(maxlen=230//2, num_hid=embed_dim-1, device = device)

    def forward(self, encoder_inputs, decoder_inputs):

        # Split inputs into real and imaginary components
        encoder_inputs_real, encoder_inputs_imag = encoder_inputs[:,:,0].unsqueeze(-1), encoder_inputs[:,:,1].unsqueeze(-1)
        decoder_inputs_real, decoder_inputs_imag = decoder_inputs[:,:,0].unsqueeze(-1), decoder_inputs[:,:,1].unsqueeze(-1)

        # Embedding
        encoder_embedded_real = self.embedding1(encoder_inputs_real)
        encoder_embedded_imag = self.embedding1(encoder_inputs_imag)
        
        decoder_embedded_real = self.embedding2(decoder_inputs_real)
        decoder_embedded_imag = self.embedding2(decoder_inputs_imag)
        
        encoder_inputs = torch.cat((encoder_embedded_real, encoder_embedded_imag), dim=-1)
        decoder_inputs = torch.cat((decoder_embedded_real, decoder_embedded_imag), dim=-1)
        

        #pass through encoder and then decoder
        encoder_outputs = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs)

        # Apply convolution to map to 2 channels (real and imaginary), output is [batch_size, seq_len, 2] 
        decoder_outputs = self.conv1d(decoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        return decoder_outputs

def create_transformer(embed_dim=128, dense_dim=64, num_heads=8, seq_length=2565, device="cpu"):
    model = TransformerModel(embed_dim, dense_dim, num_heads, seq_length, device)
    return model
    