import numpy as np
from torch import nn
from torch.nn import functional as F

from .transformer import MultiHeadAttention, MLP, EncoderBlock, Transformer


class  PositionEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, input_ch=3, embed_dim=768):
        super(PositionEmbedding, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        #stride == patch size
        self.proj = nn.Conv2d(
            in_channels=input_ch,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  

    def forward(self, x):
        x = self.proj(x)    #383 x 384 --> (n(batch size), 768(embeded dim), 24(sqrt(# patches)) ,24(sqrt(# patches)))
        x = x.flatten(2)    #(n(batch size) , 576(embeded dim),  576(# patches))
        x = x.transpose(1, 2)   #(batch size, # patches, emebed dim)
        return x

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

class VisionTransformer(nn.Module):
    def __init__(
            self, 
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_classes=1000,
            emb_size=768,
            depth=12,
            **kwargs,
            ):
        super.__init__(
            PositionEmbedding(img_size, patch_size, in_channels, emb_size),
            EncoderBlock(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )
       
