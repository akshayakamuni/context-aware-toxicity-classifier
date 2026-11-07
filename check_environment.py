import sys, platform, importlib.metadata as md
print("Python:", sys.version)
print("Platform:", platform.platform())
for package in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "tokenizers"]:
    try:
        print(f"{package}: {md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
try:
    import torch
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch check failed:", repr(e))
