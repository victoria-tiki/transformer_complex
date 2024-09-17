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
        positions = torch.arange(0, maxlen, device=x.device)
        positions = self.pos_emb(positions)
        return x + positions

class FixedEmbedding(nn.Module):
    def __init__(self, maxlen=100, num_hid=64, device="cpu"):
        super(FixedEmbedding, self).__init__()
        self.maxlen = maxlen
        self.num_hid = num_hid
        self.device = device
        self.pos_encoding = self.positional_encoding(self.maxlen, self.num_hid)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(torch.arange(position, device=self.device).unsqueeze(1),
                                     torch.arange(d_model, device=self.device).unsqueeze(0), d_model)
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads.unsqueeze(0)
        return pos_encoding

    def forward(self, x):
        bs, seq_len = x.size(0), x.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :].repeat(bs, 1, 1)
        x = torch.cat((x, pos_encoding.to(self.device)), dim=-1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, additional_heads):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.additional_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.dense_proj_additional = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.layernorm_3 = nn.LayerNorm(embed_dim)
        self.layernorm_additional = nn.LayerNorm(embed_dim)
        
    def forward(self, inputs, mask=None, additional_mask=None, return_weights=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        attention_output, attn_weights = self.attention(inputs, inputs, inputs, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        outputs = self.layernorm_2(proj_input + proj_output)
        
        if additional_mask is not None and additional_mask.any():
            additional_indices = torch.nonzero(additional_mask.squeeze(-1), as_tuple=True)[0]
            if len(additional_indices) > 0:
                additional_proj_output= self.dense_proj_additional(proj_input[additional_indices])
                outputs[additional_indices]=self.layernorm_3(proj_input[additional_indices] + additional_proj_output)
        
        if return_weights:
            return outputs, attn_weights
        else:
            return outputs


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, seq_length, additional_heads, device):
        super(TransformerDecoder, self).__init__()
        self.attention_1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.additional_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.attention_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.dense_proj_additional = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.layernorm_3 = nn.LayerNorm(embed_dim)
        self.layernorm_4 = nn.LayerNorm(embed_dim)
        self.layernorm_additional = nn.LayerNorm(embed_dim)
        self.device = device
        self.register_buffer("causal_mask", self.generate_causal_mask(seq_length))

    def generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, inputs, encoder_outputs, mask=None, additional_mask=None, return_weights=False):
        causal_mask = self.causal_mask[:inputs.size(1), :inputs.size(1)]
        if mask is not None:
            padding_mask = mask.unsqueeze(1)
            padding_mask = torch.minimum(padding_mask, causal_mask)
        else:
            padding_mask = causal_mask

        inputs = inputs.permute(1, 0, 2)
        attention_output_1, attn_weights_1 = self.attention_1(inputs, inputs, inputs, attn_mask=causal_mask,need_weights=True, average_attn_weights=False)
        attention_output_1 = attention_output_1.permute(1, 0, 2)
        attention_output_1 = self.layernorm_1(inputs.permute(1, 0, 2) + attention_output_1)

        attention_output_2, attn_weights_2 = self.attention_2(attention_output_1, encoder_outputs, encoder_outputs, key_padding_mask=None,need_weights=True, average_attn_weights=False)
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        outputs = self.layernorm_3(attention_output_2 + proj_output)
        
        if additional_mask is not None and additional_mask.any():
            additional_indices = torch.nonzero(additional_mask.squeeze(-1), as_tuple=True)[0]
            if len(additional_indices) > 0:
                additional_proj_output= self.dense_proj_additional(attention_output_2[additional_indices])
                outputs[additional_indices]=self.layernorm_4(attention_output_2[additional_indices] + additional_proj_output)

        if return_weights:
            return outputs, attn_weights_1, attn_weights_2
        else:
            return outputs

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, additional_heads, seq_length, device, return_mask):
        super(TransformerModel, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.encoder = TransformerEncoder(embed_dim * 2, dense_dim, num_heads, additional_heads)
        self.decoder = TransformerDecoder(embed_dim * 2, dense_dim, num_heads, seq_length, additional_heads, device)
        self.conv1d = nn.Conv1d(embed_dim * 2, 2, kernel_size=1, padding='same')
        self.embedding1 = FixedEmbedding(maxlen=(9900-5000)//2, num_hid=embed_dim-1, device=device)
        self.embedding2 = FixedEmbedding(maxlen=230//2, num_hid=embed_dim-1, device=device)
        self.return_mask=return_mask
        self.special_conv1d = nn.Conv1d(embed_dim * 2, 2, kernel_size=1, padding='same')

    def forward(self, encoder_inputs, decoder_inputs, return_weights=False):
        encoder_inputs_real, encoder_inputs_imag = encoder_inputs[:, :, 0].unsqueeze(-1), encoder_inputs[:, :, 1].unsqueeze(-1)
        decoder_inputs_real, decoder_inputs_imag = decoder_inputs[:, :, 0].unsqueeze(-1), decoder_inputs[:, :, 1].unsqueeze(-1)

        imag_mask = (encoder_inputs_imag <= 1e-6).all(dim=1).float().unsqueeze(-1)

        encoder_embedded_real = self.embedding1(encoder_inputs_real)
        encoder_embedded_imag = self.embedding1(encoder_inputs_imag)
        decoder_embedded_real = self.embedding2(decoder_inputs_real)
        decoder_embedded_imag = self.embedding2(decoder_inputs_imag)

        encoder_inputs = torch.cat((encoder_embedded_real, encoder_embedded_imag), dim=-1)
        decoder_inputs = torch.cat((decoder_embedded_real, decoder_embedded_imag), dim=-1)

        if return_weights:
            encoder_outputs, encoder_attn_weights = self.encoder(encoder_inputs, additional_mask=imag_mask, return_weights=True)
            decoder_outputs, decoder_self_attn_weights, cross_attn_weights = self.decoder(decoder_inputs, encoder_outputs, additional_mask=imag_mask, return_weights=True)
        else:
            encoder_outputs = self.encoder(encoder_inputs, additional_mask=imag_mask)
            decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, additional_mask=imag_mask)
        
        if imag_mask is not None and imag_mask.any():
            imag_mask_indices = torch.nonzero(imag_mask.squeeze(-1), as_tuple=True)[0]
            special_conv_outputs = self.special_conv1d(decoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)
            conv_outputs = self.conv1d(decoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)
            conv_outputs[imag_mask_indices, :, :] = special_conv_outputs[imag_mask_indices, :, :]
        else:
            conv_outputs = self.conv1d(decoder_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        decoder_outputs = conv_outputs
            
        if return_weights:
            if self.return_mask:
                return decoder_outputs, imag_mask, encoder_attn_weights, decoder_self_attn_weights, cross_attn_weights
            else:
                return decoder_outputs, encoder_attn_weights, decoder_self_attn_weights, cross_attn_weights
        else:
            if self.return_mask:
                return decoder_outputs, imag_mask
            else: 
                return decoder_outputs

def create_transformer(embed_dim=128, dense_dim=64, num_heads=8, additional_heads=8, seq_length=256, device="cpu", return_mask=False):
    model = TransformerModel(embed_dim, dense_dim, num_heads, additional_heads, seq_length, device, return_mask)
    return model
