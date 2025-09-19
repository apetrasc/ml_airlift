import torch
import torch.nn as nn
from torchvision.models import vit_b_16
class SimpleViTRegressor(nn.Module):
    def __init__(self, input_length, patch_size=50, emb_dim=128, num_layers=4, num_heads=4, mlp_dim=256, dropout=0.1):
        """
        Vision Transformer for 1D regression.
        Args:
            input_length (int): Length of the input sequence.
            patch_size (int): Size of each patch.
            emb_dim (int): Embedding dimension.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the MLP in the transformer.
            dropout (float): Dropout rate.
        """
        super(SimpleViTRegressor, self).__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        self.emb_dim = emb_dim

        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(patch_size, emb_dim)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))

        # Transformer encoder
        # If a pretrained_model is provided, use its encoder layer; otherwise, create a new one.
        if hasattr(self, 'pretrained_model') and self.pretrained_model is not None:
            encoder_layer = self.pretrained_model
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, x):
        # x: (batch, 1, length)
        batch_size = x.size(0)
        # Remove channel dimension and split into patches
        x = x.squeeze(1)  # (batch, length)
        # Ensure input length is divisible by patch_size
        if x.shape[1] % self.patch_size != 0:
            # Pad with zeros if necessary
            pad_len = self.patch_size - (x.shape[1] % self.patch_size)
            x = nn.functional.pad(x, (0, pad_len), mode='constant', value=0)
        # Reshape into patches
        x = x.unfold(1, self.patch_size, self.patch_size)  # (batch, num_patches, patch_size)
        # Project patches to embedding dimension
        x = self.patch_embed(x)  # (batch, num_patches, emb_dim)
        # Add positional embedding
        x = x + self.pos_embed
        # Pass through transformer encoder
        x = self.transformer(x)  # (batch, num_patches, emb_dim)
        # Global average pooling over patches
        x = x.mean(dim=1)  # (batch, emb_dim)
        # Regression head
        out = self.regressor(x)  # (batch, 1)
        return out.squeeze(1)  # (batch,)
