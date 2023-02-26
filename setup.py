import setuptools

setuptools.setup(
    install_requires=[
        "torch",
        "torchdiffeq",
        "torchsde",
        "omegaconf",
        "einops",
        "open-clip-torch",
        "transformers",
        "safetensors",
        "pytorch_lightning",
        "accelerate",
        "Pillow",
        "numpy",
    ],
)
