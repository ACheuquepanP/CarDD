"""CarDD · verificacion de entorno GPU (Fase 0)."""
import sys
import torch
import torchvision

EXPECTED = {"torch": "2.11.0", "torchvision": "0.26.0", "cuda": "13.0", "capability": (12, 0)}


def main() -> int:
    print(f"torch        {torch.__version__}")
    print(f"torchvision  {torchvision.__version__}")
    print(f"CUDA (build) {torch.version.cuda}")
    ok = True
    if not torch.__version__.startswith(EXPECTED["torch"]):
        print(f"  ! esperado torch {EXPECTED['torch']}"); ok = False
    if not torchvision.__version__.startswith(EXPECTED["torchvision"]):
        print(f"  ! esperado torchvision {EXPECTED['torchvision']}"); ok = False
    if torch.version.cuda != EXPECTED["cuda"]:
        print(f"  ! CUDA={torch.version.cuda}, esperado {EXPECTED['cuda']} (build cu126/cpu?)"); ok = False
    if not torch.cuda.is_available():
        print("  ! CUDA no disponible: revisa driver (>=580) y wheel cu130"); return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU          {name}")
    print(f"capability   {cap}  -> sm_{cap[0]}{cap[1]}")
    if cap != EXPECTED["capability"]:
        print(f"  ! esperado {EXPECTED['capability']} (sm_120)"); ok = False
    try:
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        print(f"kernel GPU   OK (matmul -> {y:.1f})")
    except Exception as e:
        print(f"  ! kernel GPU FALLO: {e}"); return 1
    print("\nOK - entorno listo" if ok else "\nFALLA - revisa los '!'")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
