import torch
import torch.nn as nn
from typing import Dict, Tuple
import pytorch_lightning as L
import torch.nn.functional as F


import torch
import torch.nn as nn

shape = (200, 200)
class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim, input_shape, n_encoder_layers=4, channels_base=16, dropout=0.5):
        super(ConvAutoEncoder, self).__init__()
        self.__version__ = "v1.0.1"
        self.input_shape = input_shape  # (height, width)
        self.n_encoder_layers = n_encoder_layers
        self.channels_base = channels_base

        # Build encoder and track spatial dimensions
        self.encoder, self.encoder_shapes = self._build_encoder(dropout)

        # Calculate flattened size after encoder
        last_channels, last_height, last_width = self.encoder_shapes[-1]
        self.flattened_size = last_channels * last_height * last_width

        # Latent space layers
        self.fc_enc = nn.Sequential(
            nn.Linear(self.flattened_size, latent_dim),
            nn.LeakyReLU(True),
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.LeakyReLU(True),
        )

        # Build decoder
        self.decoder = self._build_decoder(dropout)

    def _build_encoder(self, dropout):
        layers = []
        current_channels = 1  # Input channels
        current_height, current_width = self.input_shape
        shapes = [(current_channels, current_height, current_width)]

        for layer_idx in range(self.n_encoder_layers):
            out_channels = self.channels_base * (2 ** layer_idx)

            conv_block = [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(True)
            ]

            # Add dropout to all layers except last
            if layer_idx != self.n_encoder_layers - 1:
                conv_block.append(nn.Dropout2d(dropout))

            layers.extend(conv_block)

            # Update spatial dimensions
            current_height = (current_height + 2*1 - 3) // 2 + 1
            current_width = (current_width + 2*1 - 3) // 2 + 1
            current_channels = out_channels
            shapes.append((current_channels, current_height, current_width))

        return nn.Sequential(*layers), shapes

    def _build_decoder(self, dropout):
        layers = []
        decoder_shapes = self.encoder_shapes[::-1][:-1]  # Reverse and exclude input shape

        current_channels, current_h, current_w = decoder_shapes[0]

        for layer_idx in range(self.n_encoder_layers):
            # Determine target shape
            if layer_idx < self.n_encoder_layers - 1:
                next_channels, target_h, target_w = decoder_shapes[layer_idx+1]
            else:
                next_channels = 1
                target_h, target_w = self.input_shape

            # Calculate output padding
            output_padding_h = target_h - ((current_h - 1)*2 - 2*1 + 3)
            output_padding_w = target_w - ((current_w - 1)*2 - 2*1 + 3)
            output_padding = output_padding_h  # Assuming square images

            # Create transpose conv layer
            conv_t = nn.ConvTranspose2d(
                current_channels, next_channels,
                kernel_size=3, stride=2, padding=1,
                output_padding=output_padding
            )
            layers.append(conv_t)

            # Add activations and dropout
            if layer_idx != self.n_encoder_layers - 1:
                layers.append(nn.LeakyReLU(True))
                layers.append(nn.Dropout2d(dropout))
            else:
                layers.append(nn.Sigmoid())

            current_channels = next_channels
            current_h, current_w = target_h, target_w

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, h, w = x.size()
        x = x.view(batch_size, 1, h, w)

        # Encode
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.fc_enc(x)

        # Decode
        x = self.fc_dec(x)
        x = x.view(batch_size, *self.encoder_shapes[-1])
        x = self.decoder(x)

        # Reshape back to original format
        x = x.view(batch_size, *self.input_shape)
        return x

    def encode(self, x):
        batch_size, h, w = x.size()
        x = x.view(batch_size, 1, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        return self.fc_enc(x).view(batch_size, -1)

    def decode(self, z):
        batch_size, latent_dim = z.size()
        z = z.view(batch_size, latent_dim)
        z = self.fc_dec(z)
        z = z.view(batch_size, *self.encoder_shapes[-1])
        return self.decoder(z).view(batch_size, *self.input_shape)


import lightning.pytorch as pl


class LitConvAutoEncoder(pl.LightningModule):
    def __init__(self, latent_dim, n_encoder_layers, channels_base, lr=1e-3, weight_decay=1e-5, dropout=0.5):
        super(LitConvAutoEncoder, self).__init__()
        self.save_hyperparameters()
        self.model = ConvAutoEncoder(
            input_shape=(200, 200),
            latent_dim=latent_dim,
            n_encoder_layers=n_encoder_layers,
            channels_base=channels_base,
            dropout=dropout
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss, prog_bar=True)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [lr_sched]
        #return optimizer


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        input = self.dropout(input)
        output, (hidden, cell) = self.lstm(input)

        return output, (hidden, cell)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, decoder_inputs):
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(decoder_inputs.shape[1]):
            decoder_input = decoder_inputs[:, i].unsqueeze(1)
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        query = hidden[0].permute(1, 0, 2)
        query = self.dropout(query)

        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((input, context), dim=2)

        output, hidden = self.lstm(input_lstm, hidden)
        output = self.out(output) + input

        return output, hidden, attn_weights

class LitSeq2SeqModel(L.LightningModule):
    def __init__(self, encoder, decoder):
        super(LitSeq2SeqModel, self).__init__()
        self.__version__ = "v1.0.0"
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, y):
        encoder_input = X[0] # only use the high fidelity data
        y_coarse = y[1] # only use the low fidelity data as input to the decoder
        encoder_outputs, encoder_hidden = self.encoder(encoder_input)
        decoder_outputs, _, attentions = self.decoder(encoder_outputs, encoder_hidden, y_coarse)

        return decoder_outputs, attentions

    def training_step(self, batch, _):
        Z, y = batch

        decoder_outputs, attentions = self(Z, y)

        loss = F.mse_loss(decoder_outputs, y[0])
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        Z, y = batch

        decoder_outputs, attentions = self(Z, y)

        loss = F.mse_loss(decoder_outputs, y[0])
        self.log('val_loss', loss, prog_bar=True)
        # self.log('attention', attentions)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
