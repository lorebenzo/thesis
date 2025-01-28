import torch
import torch.nn as nn
from typing import Dict, Tuple

class ConvAutoEncoder(nn.Module):
    """Convolutional Autoencoder with configurable architecture.
    
    Args:
        config: Dictionary containing model configuration parameters
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.__version__ = '1.0.0'
        arch_config = config['architecture']
        self.shape = arch_config['shape']
        self.latent_dim = arch_config['latent_dim']
        
        # Build model components
        self.encoder = self._build_encoder(arch_config)
        self.decoder = self._build_decoder(arch_config)
        
        # Calculate encoder output shape dynamically
        self.encoder_output_shape = self._compute_encoder_output_shape()
        self.encoder_output_size = 128 * self.encoder_output_shape[0] * self.encoder_output_shape[1]
        
        # Build latent space projection layers
        self.fc_enc = nn.Sequential(
            nn.Linear(self.encoder_output_size, self.latent_dim),
            nn.LeakyReLU(inplace=True),
        )
        
        self.fc_dec = nn.Sequential(
            nn.Linear(self.latent_dim, self.encoder_output_size),
            nn.LeakyReLU(inplace=True),
        )

    def _build_encoder(self, config: Dict) -> nn.Sequential:
        """Construct the encoder network."""
        layers = []
        in_channels = 1
        out_channels = [16, 32, 64, 128]
        
        for channels in out_channels:
            layers += [
                nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(inplace=True),
                nn.Dropout2d(config['dropout_p']),
            ]
            in_channels = channels
            
        return nn.Sequential(*layers[:-1])  # Remove last dropout

    def _build_decoder(self, config: Dict) -> nn.Sequential:
        """Construct the decoder network."""
        layers = []
        in_channels = 128
        out_channels = [64, 32, 16, 1]
        output_paddings = [0, 1, 1, 1]
        
        for channels, out_pad in zip(out_channels, output_paddings):
            layers += [
                nn.ConvTranspose2d(in_channels, channels, 
                                 kernel_size=3, stride=2, 
                                 padding=1, output_padding=out_pad),
                nn.LeakyReLU(inplace=True),
                nn.Dropout2d(config['dropout_p']),
            ]
            in_channels = channels
            
        # Final layer with sigmoid activation
        layers[-1] = nn.Sigmoid()
        return nn.Sequential(*layers)

    def _compute_encoder_output_shape(self) -> Tuple[int, int]:
        """Calculate encoder output shape based on input dimensions."""
        h, w = self.shape
        for _ in range(4):  # 4 conv layers with stride 2
            h = (h + 1) // 2
            w = (w + 1) // 2
        return h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, height, width)
            
        Returns:
            Reconstructed tensor of same shape as input
        """
        batch_size = x.size(0)
        
        # Add channel dimension and encode
        x = x.view(batch_size, 1, *self.shape)
        x = self.encoder(x)
        
        # Flatten and project to latent space
        x = x.view(batch_size, -1)
        x = self.fc_enc(x)
        
        # Project back and decode
        x = self.fc_dec(x)
        x = x.view(batch_size, 128, *self.encoder_output_shape)
        x = self.decoder(x)
        
        # Remove channel dimension
        return x.view(batch_size, *self.shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space representation."""
        batch_size = x.size(0)
        x = x.view(batch_size, 1, *self.shape)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        return self.fc_enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to input space."""
        batch_size = z.size(0)
        x = self.fc_dec(z)
        x = x.view(batch_size, 128, *self.encoder_output_shape)
        x = self.decoder(x)
        return x.view(batch_size, *self.shape)
    
    
__all__ = ["ConvAutoEncoder"]  # Explicitly list exported names