# Original User Request

## 2026-09-14T03:27:06Z

This is a single self-contained fix; keep it small and focused.

Implement an Illumination-Guided Delta Modulation Mamba module (Method 1) for the HV branch in the Low-Light Image Enhancement (HVI-CIDNet) project, where features from branch I act as an accelerator/brake controlling the step size $\Delta$ of Mamba scanning over branch HV, eliminating chromatic noise and stabilizing hidden states.

Working directory: E:/PythonFile/Project/Low-Light-Image-Enhancement

## Requirements

### R1. Illumination-Guided Selective Scan Block (IG-Mamba)
Implement a standalone PyTorch module (`src/HVI-CIDNet/net/IG_Mamba.py`) that accepts two inputs: $I$ (intensity feature map from branch I) and $HV$ (chrominance feature map from branch HV). The module must compute the selective scan step size $\Delta$ by conditioning on $I$, effectively damping $\Delta \to 0$ in dark/noisy regions.

### R2. Seamless Dimension & Spatial Preservation
Ensure the module preserves spatial dimensions, supports 4-direction scanning (or SS2D 2D selective scan), maintains tensor compatibility with the CIDNet UNet architecture, and avoids duplicating sequence length.

### R3. Comprehensive Code Explanation & Logic Verification
Provide a step-by-step mathematical and algorithmic explanation of every component, including a unit test script to verify forward pass, gradient flow, and edge cases (e.g. extreme dark inputs where $I \approx 0$).

## Acceptance Criteria

### Correctness & Logic Integrity
- [ ] Tensor shapes for $I$ and $HV$ matching $[B, C, H, W]$ produce an output $HV_{\text{out}}$ of identical shape $[B, C, H, W]$ without dimension errors.
- [ ] In extremely dark regions ($I \to 0$), the modulation factor strongly attenuates $\Delta_t$, preventing noise from polluting the hidden states $h_t$.
- [ ] Gradient backward pass runs cleanly without NaN or infinite gradients.
- [ ] Verified compatibility with CUDA / PyTorch without breaking spatial locality.
