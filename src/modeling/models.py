import torch
import torch.nn as nn
from typing import Dict, Tuple
import pytorch_lightning as L
import torch.nn.functional as F


import torch
import torch.nn as nn

shape = (200, 200)


class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim, shape):
        super(ConvAutoEncoder, self).__init__()
        self.__version__ = "v1.0.0"
        self.shape = shape

        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1
            ),  # Output: (batch_size, 16, 50, 50)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Dropout2d(0.5),
            nn.Conv2d(
                16, 32, kernel_size=3, stride=2, padding=1
            ),  # Output: (batch_size, 32, 25, 25)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Dropout2d(0.5),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # Output: (batch_size, 64, 13, 13)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Dropout2d(0.5),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # Output: (batch_size, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
        )

        # Flatten for FC layers
        self.flatten = nn.Flatten()

        # FC Layers
        self.fc_enc = nn.Sequential(
            nn.Linear(128 * 13 * 13, latent_dim),
            nn.LeakyReLU(True),
        )

        self.fc_dec = nn.Sequential(nn.Linear(latent_dim, 128 * 13 * 13), nn.LeakyReLU(True))

        # CNN Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(True),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Assuming input shape: (batch_size, sequence_length, 1, 100, 100) for 100x100 images

        batch_size, _, _ = x.size()

        # Encode each time step separately
        x = x.view(batch_size, 1, *shape)  # Combine batch and sequence dimensions
        x = self.encoder(x)

        x = self.flatten(x)

        x = self.fc_enc(x)

        x = self.fc_dec(x)

        x = x.view(batch_size, 128, 13, 13)

        # Decode
        x = self.decoder(x)  # Output: (batch_size, 1, 100, 100)

        x = x.view(batch_size, *shape)  # Split batch and sequence dimensions

        return x

    def decode(self, x):
        (
            batch_size,
            _,
        ) = x.size()
        x = self.fc_dec(x)
        x = x.view(batch_size, 128, 13, 13)
        x = self.decoder(x)
        x = x.view(batch_size, *shape)

        return x

    def encode(self, x):
        batch_size, _, _ = x.size()

        # Encode each time step separately
        x = x.view(batch_size, 1, *shape)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_enc(x)

        return x


class LitConvAutoEncoder(L.LightningModule):
    def __init__(self, autoencoder, lr=1e-3, weight_decay=1e-5, step_size=10, gamma=0.1):
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        self.save_hyperparameters(ignore=["autoencoder"])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.autoencoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.autoencoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)  # Optional: Log per batch if needed
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.autoencoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", loss, prog_bar=True)

    def forward(self, x):
        return self.autoencoder(x)


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

        for i in range(80):
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