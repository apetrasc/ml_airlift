#!/usr/bin/env python3
"""
CNN models specifically designed for image-based signal data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageSignalCNN(nn.Module):
    """CNN model for image-based signal data."""
    
    def __init__(self, in_channels: int = 3, out_dim: int = 6, 
                 backbone: str = 'resnet18', pretrained: bool = True,
                 dropout_rate: float = 0.5):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            out_dim: Number of output dimensions
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50', 'custom')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(ImageSignalCNN, self).__init__()
        
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.backbone = backbone
        self.dropout_rate = dropout_rate
        
        if backbone == 'resnet18':
            self.backbone_model = models.resnet18(pretrained=pretrained)
            self.backbone_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone_model.fc = nn.Linear(self.backbone_model.fc.in_features, out_dim)
            
        elif backbone == 'resnet34':
            self.backbone_model = models.resnet34(pretrained=pretrained)
            self.backbone_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone_model.fc = nn.Linear(self.backbone_model.fc.in_features, out_dim)
            
        elif backbone == 'resnet50':
            self.backbone_model = models.resnet50(pretrained=pretrained)
            self.backbone_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone_model.fc = nn.Linear(self.backbone_model.fc.in_features, out_dim)
            
        elif backbone == 'custom':
            self.backbone_model = self._build_custom_backbone()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Add dropout if specified
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def _build_custom_backbone(self):
        """Build custom CNN backbone."""
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(self.in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 1
            self._make_layer(32, 64, 2, stride=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.out_dim)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Make a layer with residual blocks."""
        layers = []
        
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        x = self.backbone_model(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for custom backbone."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class EfficientNetSignalCNN(nn.Module):
    """EfficientNet-based model for signal images."""
    
    def __init__(self, in_channels: int = 3, out_dim: int = 6, 
                 model_size: str = 'b0', pretrained: bool = True):
        """
        Args:
            in_channels: Number of input channels
            out_dim: Number of output dimensions
            model_size: EfficientNet model size ('b0', 'b1', 'b2', etc.)
            pretrained: Whether to use pretrained weights
        """
        super(EfficientNetSignalCNN, self).__init__()
        
        try:
            import torchvision.models as models
            
            # Load EfficientNet model
            if model_size == 'b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
            elif model_size == 'b1':
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
            elif model_size == 'b2':
                self.backbone = models.efficientnet_b2(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported EfficientNet size: {model_size}")
            
            # Modify first layer for different input channels
            if in_channels != 3:
                self.backbone.features[0][0] = nn.Conv2d(
                    in_channels, self.backbone.features[0][0].out_channels,
                    kernel_size=self.backbone.features[0][0].kernel_size,
                    stride=self.backbone.features[0][0].stride,
                    padding=self.backbone.features[0][0].padding,
                    bias=False
                )
            
            # Modify classifier
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Linear(in_features, out_dim)
            
        except ImportError:
            print("EfficientNet not available, falling back to ResNet18")
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim)
    
    def forward(self, x):
        return self.backbone(x)


class VisionTransformerSignalCNN(nn.Module):
    """Vision Transformer for signal images."""
    
    def __init__(self, in_channels: int = 3, out_dim: int = 6, 
                 patch_size: int = 16, img_size: int = 224,
                 hidden_dim: int = 768, num_heads: int = 12, 
                 num_layers: int = 12, pretrained: bool = True):
        """
        Args:
            in_channels: Number of input channels
            out_dim: Number of output dimensions
            patch_size: Size of image patches
            img_size: Input image size
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            pretrained: Whether to use pretrained weights
        """
        super(VisionTransformerSignalCNN, self).__init__()
        
        try:
            import torchvision.models as models
            
            # Load Vision Transformer
            self.backbone = models.vit_b_16(pretrained=pretrained)
            
            # Modify for different input channels
            if in_channels != 3:
                # Get the patch embedding layer
                patch_embed = self.backbone.conv_proj
                self.backbone.conv_proj = nn.Conv2d(
                    in_channels, patch_embed.out_channels,
                    kernel_size=patch_embed.kernel_size,
                    stride=patch_embed.stride,
                    padding=patch_embed.padding
                )
            
            # Modify head
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, out_dim)
            
        except (ImportError, AttributeError):
            print("Vision Transformer not available, falling back to ResNet18")
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim)
    
    def forward(self, x):
        return self.backbone(x)


def create_model(model_type: str = 'resnet18', in_channels: int = 3, 
                out_dim: int = 6, **kwargs) -> nn.Module:
    """Factory function to create models."""
    
    if model_type.startswith('resnet'):
        return ImageSignalCNN(
            in_channels=in_channels,
            out_dim=out_dim,
            backbone=model_type,
            **kwargs
        )
    elif model_type.startswith('efficientnet'):
        return EfficientNetSignalCNN(
            in_channels=in_channels,
            out_dim=out_dim,
            model_size=model_type.split('_')[1] if '_' in model_type else 'b0',
            **kwargs
        )
    elif model_type.startswith('vit'):
        return VisionTransformerSignalCNN(
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test different model types
    model_types = ['resnet18', 'resnet34', 'efficientnet_b0', 'vit_b_16']
    
    for model_type in model_types:
        try:
            model = create_model(model_type, in_channels=3, out_dim=6)
            print(f"✅ {model_type}: {sum(p.numel() for p in model.parameters())} parameters")
        except Exception as e:
            print(f"❌ {model_type}: {e}")
    
    # Test with different input channels
    print("\nTesting different input channels...")
    for channels in [1, 3, 4]:
        try:
            model = create_model('resnet18', in_channels=channels, out_dim=6)
            print(f"✅ ResNet18 with {channels} channels: {sum(p.numel() for p in model.parameters())} parameters")
        except Exception as e:
            print(f"❌ ResNet18 with {channels} channels: {e}")




