
def select_configs(arch):
    if arch == 'B_16':
        return get_base_config()
    else:
        raise NotImplementedError(f"Unknown model '{arch}'.")

def get_base_config():
    """
    ViT-B/16 Imagenet1K
    """
    return dict({
        'image_size': 224,
        'patch_size': 16,
        'num_heads': 12,
        'num_layers': 12,
        'embed_size': 768,
        'mlp_size': 3072,
        'dropout_rate': 0.1,
        'num_classes': 1000
    })
