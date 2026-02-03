import platform
import sys


def must_import(name: str):
    try:
        return __import__(name)
    except Exception as e:
        print(f"[FAIL] Could not import '{name}': {e}")
        sys.exit(1)


def main():
    print("=== Environment Check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    pandas = must_import("pandas")
    sklearn = must_import("sklearn")
    torch = must_import("torch")

    print("\n=== Package Versions ===")
    print(f"pandas: {getattr(pandas, '__version__', 'unknown')}")
    print(f"scikit-learn: {getattr(sklearn, '__version__', 'unknown')}")
    print(f"torch: {getattr(torch, '__version__', 'unknown')}")

    # Accelerator detection
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    print("\n=== Accelerator Status ===")
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    print(f"MPS available: {mps_ok}")

    if cuda_ok:
        device = torch.device("cuda")
    elif mps_ok:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Selected device: {device}")

    # Tensor computation
    print("\n=== Tensor Compute Test ===")
    try:
        a = torch.randn((256, 256), device=device)
        b = torch.randn((256, 256), device=device)
        c = a @ b  # matrix multiply

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("[OK] Tensor matmul succeeded.")
        print(f"Result: shape={tuple(c.shape)}, dtype={c.dtype}, device={c.device}")
        print(f"Checksum (mean): {c.mean().item():.6f}")
    except Exception as e:
        print(f"[FAIL] Tensor compute failed on {device}: {e}")
        sys.exit(1)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
