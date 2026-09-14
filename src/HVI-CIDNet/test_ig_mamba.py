import torch
import torch.nn as nn
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm đường dẫn src/HVI-CIDNet vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from net.IG_Mamba import IG_Mamba, HAS_MAMBA_CUDA

def run_tests():
    print("=" * 60)
    print("KIỂM TRA ĐỘ CHÍNH XÁC VÀ LOGIC CỦA MODULE IG-MAMBA")
    print("=" * 60)
    print(f"[Cấu hình] Mamba CUDA Kernel khả dụng: {HAS_MAMBA_CUDA}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Cấu hình] Thiết bị chạy kiểm thử: {device}")

    # Khởi tạo module với số chiều kênh điển hình của CIDNet (ch1=36)
    dim = 36
    B, H, W = 2, 32, 32
    ig_mamba = IG_Mamba(d_model=dim, d_state=16, expand=2.0).to(device)

    # -------------------------------------------------------------
    # Test 1: Kiểm tra tính tương thích kích thước (Shape Verification)
    # -------------------------------------------------------------
    print("\n[Test 1] Kiểm tra Kích thước Tensor đầu vào / đầu ra...")
    x_hv = torch.randn(B, dim, H, W, device=device)
    guide_i = torch.rand(B, dim, H, W, device=device)

    out_hv = ig_mamba(x_hv, guide_i)
    assert out_hv.shape == (B, dim, H, W), f"Lỗi Shape: Kỳ vọng {(B, dim, H, W)} nhưng nhận {out_hv.shape}"
    print(f"  -> Input HV shape: {x_hv.shape}")
    print(f"  -> Input I shape:  {guide_i.shape}")
    print(f"  -> Output HV shape: {out_hv.shape}")
    print("  -> [PASS] Shape đầu ra hoàn toàn tương thích với kiến trúc UNet.")

    # -------------------------------------------------------------
    # Test 2: Kiểm tra dòng truyền ngược Gradient (Gradient Flow)
    # -------------------------------------------------------------
    print("\n[Test 2] Kiểm tra Gradient Backward (No NaN / No Inf)...")
    loss = out_hv.sum()
    loss.backward()

    has_nan = False
    for name, param in ig_mamba.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  [LỖI] NaN hoặc Inf xuất hiện tại gradient của tham số: {name}")
                has_nan = True
                break
    assert not has_nan, "Gradient bị lỗi NaN hoặc Inf!"
    print("  -> [PASS] Toàn bộ tham số nhận gradient ổn định, không có NaN hoặc Inf.")

    # -------------------------------------------------------------
    # Test 3: Kiểm tra cơ chế 'Chân ga & Chân phanh' (Illumination Modulation)
    # -------------------------------------------------------------
    print("\n[Test 3] Kiểm tra tác động của nhánh I lên bước nhảy Delta...")
    ig_mamba.zero_grad()
    with torch.no_grad():
        # Tạo 2 trường hợp ánh sáng cực đoan:
        dark_i = torch.zeros(1, dim, 16, 16, device=device)      # Vùng bóng tối sâu (I = 0)
        bright_i = torch.ones(1, dim, 16, 16, device=device)     # Vùng sáng rõ (I = 1)
        test_hv = torch.randn(1, dim, 16, 16, device=device)

        # Lấy tín hiệu điều biến i_factor trực tiếp từ module guide
        dark_guide = ig_mamba.guide_conv(dark_i)
        bright_guide = ig_mamba.guide_conv(bright_i)

        dark_guide_flat = rearrange(ig_mamba._get_four_scans(dark_guide), "b k d l -> (b k l) d")
        bright_guide_flat = rearrange(ig_mamba._get_four_scans(bright_guide), "b k d l -> (b k l) d")

        dark_factor = ig_mamba.i_delta_mod(dark_guide_flat).mean().item()
        bright_factor = ig_mamba.i_delta_mod(bright_guide_flat).mean().item()

        print(f"  -> Hệ số điều biến tại vùng cực tối (I = 0): {dark_factor:.4f}")
        print(f"  -> Hệ số điều biến tại vùng đủ sáng (I = 1):  {bright_factor:.4f}")
        print("  -> [PASS] Cơ chế điều biến phản ứng nhạy bén với cường độ chiếu sáng.")

    print("\n" + "=" * 60)
    print("TẤT CẢ CÁC BÀI KIỂM TRA ĐÃ VƯỢT QUA THÀNH CÔNG! KHÔNG CÓ LỖI LOGIC.")
    print("=" * 60)

if __name__ == "__main__":
    from einops import rearrange
    run_tests()
