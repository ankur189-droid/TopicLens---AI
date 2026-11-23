
import torch

print("=" * 50)
print("TOPICLENS - GPU AVAILABILITY CHECK")
print("=" * 50)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))

    test_tensor = torch.randn(3, 3).to("cuda")
    print("\nGPU tensor test:")
    print(test_tensor)

    print("\nGPU is ready for embedding generation.")
else:
    print("\nCUDA is not available.")
    print("BERTopic can still run using CPU.")

print("=" * 50)
