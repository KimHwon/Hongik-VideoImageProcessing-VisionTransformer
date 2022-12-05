
def get_base_config():
    """
    ViT-B/16 Imagenet1K
    """
    return dict({
        'config': {
            'dim': 768,
            'dropout_rate': 0.1,
            'ff_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'patches': 16,
        },
        'image_size': 224,
        'num_classes': 1000
    })
