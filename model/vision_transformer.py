import torch
from torch import nn

from .transformer import Transformer


class Embedding(nn.Module):
    """
    Embedding
    """
    def __init__(self, img_size, patch_size, input_ch=3, embed_dim=768):
        super().__init__()

        # Patch embedding
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            in_channels=input_ch,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  

        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.n_patches += 1

        # Positional embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

    def forward(self, x):
        b, c, h, w = x.shape

        # Patch embedding
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        # Class token
        class_tokens = self.class_token.expand(b, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        # Positional embedding
        x = x + self.position_embedding

        return x

class ClassificationHead(nn.Module):
    """
    ClassificationHead
    """
    def __init__(self, emb_size=768, n_classes=10):
        super().__init__()

        self.norm = nn.LayerNorm(emb_size),
        self.fc = nn.Linear(emb_size, n_classes)
    
    def forward(self, x):
        x = self.norm(x)[:, 0]
        x = self.fc(x)
        return x

class VisionTransformer(nn.Sequential):
    """
    VisionTransformer
    """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            n_classes: int = 1000,
            emb_size: int = 768,
            mlp_size: int = 3072,
            num_heads: int = 12,
            depth: int = 12,
            dropout_rate: float = 0.1
        ):
        super().__init__(
            Embedding(img_size, patch_size, in_channels, emb_size),
            Transformer(depth, emb_size, num_heads, mlp_size, dropout_rate),
            ClassificationHead(emb_size, n_classes)
        )
       
