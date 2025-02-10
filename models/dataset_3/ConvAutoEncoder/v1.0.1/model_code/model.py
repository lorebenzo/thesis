import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim, input_shape, n_encoder_layers=4, channels_base=16, dropout=0.5):
        super(ConvAutoEncoder, self).__init__()
        self.__version__ = "v1.0.0"
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
        batch_size, seq_len, h, w = x.size()
        x = x.view(batch_size * seq_len, 1, h, w)
        x = self.encoder(x)
        x = x.view(batch_size * seq_len, -1)
        return self.fc_enc(x).view(batch_size, seq_len, -1)

    def decode(self, z):
        batch_size, seq_len, latent_dim = z.size()
        z = z.view(batch_size * seq_len, latent_dim)
        z = self.fc_dec(z)
        z = z.view(batch_size * seq_len, *self.encoder_shapes[-1])
        return self.decoder(z).view(batch_size, seq_len, *self.input_shape)