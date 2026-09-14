import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ---------------------------------------------------------------------------
# Tự động phát hiện và import selective_scan_fn từ mamba_ssm.
# Nếu môi trường chưa cài đặt kernel CUDA (ví dụ trên Windows),
# module sẽ tự động chuyển sang pure-PyTorch implementation để không bao giờ bị crash.
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False

def selective_scan_ref_pytorch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
    delta_bias: torch.Tensor = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Hiện thực hóa thuật toán Selective Scan (SSM) thuần PyTorch làm fallback.
    u: [B, D, L]
    delta: [B, D, L]
    A: [D, N]
    B: [B, N, L] hoặc [B, G, N, L]
    C: [B, N, L] hoặc [B, G, N, L]
    D: [D]
    delta_bias: [D]
    """
    b, d, l = u.shape
    n = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Đảm bảo B, C có đúng chiều [B, N, L]
    if B.dim() == 4:
        B = B.squeeze(1)
    if C.dim() == 4:
        C = C.squeeze(1)

    # Tính toán ma trận chuyển đổi trạng thái liên tục sang rời rạc
    # delta: [B, D, L], A: [D, N] -> deltaA: [B, D, L, N]
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    
    # deltaB: [B, D, L, N]
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    # Vòng lặp quét đệ quy qua L bước (Recurrent Selective Scan)
    ys = []
    x_state = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    for i in range(l):
        x_state = deltaA[:, :, i, :] * x_state + deltaB_u[:, :, i, :]
        y_i = torch.einsum('bdn,bn->bd', x_state, C[:, :, i])
        ys.append(y_i)

    y = torch.stack(ys, dim=-1) # [B, D, L]

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)

    return y


class IlluminationGuidedDeltaMamba(nn.Module):
    """
    IG-Mamba: Nhánh I điều khiển bước nhảy Delta (Chân ga / Chân phanh) cho Mamba nhánh HV.
    
    Cơ chế:
        - Nhánh HV cung cấp đặc trưng màu sắc cần được khử nhiễu.
        - Nhánh I (đã qua Cross-Attention) đóng vai trò tín hiệu chiếu sáng tin cậy.
        - Nhánh I trực tiếp điều biến delta (dt) trong Selective Scan:
            + Ở vùng sáng rõ: delta lớn -> nạp thông tin màu sắc bình thường.
            + Ở vùng tối sâu/nhiễu: delta -> 0 -> khóa cổng nạp nhiễu, trạng thái ẩn h_t
              giữ nguyên bối cảnh mượt mà từ trước đó, triệt tiêu hiện tượng kéo vệt màu.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # --- 1. Tiền xử lý nhánh HV ---
        self.in_proj_hv = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d_hv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act_hv = nn.SiLU()

        # --- 2. Trích xuất tín hiệu hướng dẫn từ nhánh I ---
        # Học bản đồ chiếu sáng cục bộ từ nhánh I để làm chân ga/chân phanh
        self.guide_conv = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_inner, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1, bias=False)
        )

        # --- 3. Tham số SSM cho 4 hướng quét (K = 4) ---
        self.K = 4

        # Chiếu x_hv thành [dt_rank + 2 * d_state]
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # Bộ điều biến bước nhảy Delta từ nhánh I
        # Tanh đưa giá trị về [-1, 1] để kiểm soát việc tăng tốc hay phanh lại
        self.i_delta_mod = nn.Sequential(
            nn.Linear(self.d_inner, self.dt_rank, bias=True),
            nn.Tanh()
        )

        # Ma trận chiếu dt_rank lên d_inner
        self.dt_projs = nn.Parameter(torch.empty(self.K, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(self.K, self.d_inner))
        self._init_dt_projs(dt_scale, dt_init, dt_min, dt_max, dt_init_floor)

        # Khởi tạo tham số S4D cho A và D
        self.A_logs = self._init_A_logs(self.d_state, self.d_inner, copies=self.K)
        self.Ds = self._init_Ds(self.d_inner, copies=self.K)

        # --- 4. Hậu xử lý & Gating đầu ra ---
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def _init_dt_projs(self, dt_scale, dt_init, dt_min, dt_max, dt_init_floor):
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_projs, dt_init_std)
        else:
            nn.init.uniform_(self.dt_projs, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.K, self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt)

    def _init_A_logs(self, d_state, d_inner, copies):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        A_log = repeat(A_log, "d n -> r d n", r=copies)
        return nn.Parameter(A_log.flatten(0, 1))

    def _init_Ds(self, d_inner, copies):
        D = torch.ones(d_inner)
        D = repeat(D, "d -> r d", r=copies)
        return nn.Parameter(D.flatten(0, 1))

    def _get_four_scans(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> Trả về 4 chuỗi quét [B, 4, C, L] với L = H * W
        """
        B, C, H, W = x.shape
        L = H * W
        scan1 = x.view(B, C, L)
        scan2 = x.flip(-1).flip(-2).view(B, C, L)
        scan3 = x.transpose(-1, -2).contiguous().view(B, C, L)
        scan4 = x.transpose(-1, -2).flip(-1).flip(-2).contiguous().view(B, C, L)
        return torch.stack([scan1, scan2, scan3, scan4], dim=1)

    def _merge_four_scans(self, y_scans: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        y_scans: [B, 4, C, L] -> Khôi phục về [B, C, H, W]
        """
        B, _, C, L = y_scans.shape
        y1 = y_scans[:, 0].view(B, C, H, W)
        y2 = y_scans[:, 1].view(B, C, H, W).flip(-1).flip(-2)
        y3 = y_scans[:, 2].view(B, C, W, H).transpose(-1, -2)
        y4 = y_scans[:, 3].view(B, C, W, H).flip(-1).flip(-2).transpose(-1, -2)
        return y1 + y2 + y3 + y4

    def forward(self, x_hv: torch.Tensor, guide_i: torch.Tensor) -> torch.Tensor:
        """
        x_hv:    [B, C, H, W]
        guide_i: [B, C, H, W]
        """
        B, C, H, W = x_hv.shape
        L = H * W

        # 1. Chiếu kênh nhánh HV
        x_hv_perm = rearrange(x_hv, "b c h w -> b h w c")
        xz_hv = self.in_proj_hv(x_hv_perm)
        x_branch, z_branch = xz_hv.chunk(2, dim=-1)

        x_branch = rearrange(x_branch, "b h w c -> b c h w").contiguous()
        x_branch = self.act_hv(self.conv2d_hv(x_branch))

        # 2. Chiếu đặc trưng dẫn đường nhánh I
        guide_feat = self.guide_conv(guide_i) # [B, d_inner, H, W]

        # 3. Tạo 4 chuỗi quét độ dài HW (tiết kiệm 50% so với cơ chế xen kẽ cũ)
        xs = self._get_four_scans(x_branch)       # [B, 4, d_inner, L]
        guides = self._get_four_scans(guide_feat) # [B, 4, d_inner, L]

        # 4. Tính toán tham số SSM & Illumination-Guided Modulation
        xs_trans = rearrange(xs, "b k d l -> (b k l) d")
        guides_trans = rearrange(guides, "b k d l -> (b k l) d")

        # Base delta, B, C từ nhánh HV
        x_dbl = self.x_proj(xs_trans)
        dt_base, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Chân ga / chân phanh từ nhánh I
        i_factor = self.i_delta_mod(guides_trans) # [-1, 1]

        # Điều biến dt: Vùng tối ép dt giảm mạnh, vùng sáng tăng độ nhạy
        dt_modulated = dt_base + i_factor
        dt_modulated = rearrange(dt_modulated, "(b k l) r -> b k l r", b=B, k=self.K, l=L)

        # Chiếu dt lên d_inner
        dt_projected = torch.einsum("b k l r, k d r -> b k d l", dt_modulated, self.dt_projs)

        # Chuẩn bị định dạng cho scan
        xs_scan = xs.float().view(B, -1, L)
        dts_scan = dt_projected.float().contiguous().view(B, -1, L)
        Bs_scan = rearrange(B_mat, "(b k l) s -> b k s l", b=B, k=self.K, l=L).float()
        Cs_scan = rearrange(C_mat, "(b k l) s -> b k s l", b=B, k=self.K, l=L).float()

        A_mat = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        D_vec = self.Ds.float().view(-1)
        bias_vec = self.dt_projs_bias.float().view(-1)

        # 5. Thực thi Selective Scan
        if HAS_MAMBA_CUDA and xs_scan.is_cuda:
            out_y = selective_scan_fn(
                xs_scan,
                dts_scan,
                A_mat,
                Bs_scan,
                Cs_scan,
                D_vec,
                z=None,
                delta_bias=bias_vec,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, self.K, self.d_inner, L)
        else:
            # Fallback thuần PyTorch cho môi trường không có Mamba CUDA kernel
            out_scans = []
            for k_idx in range(self.K):
                u_k = xs_scan[:, k_idx * self.d_inner : (k_idx + 1) * self.d_inner, :]
                dt_k = dts_scan[:, k_idx * self.d_inner : (k_idx + 1) * self.d_inner, :]
                A_k = A_mat[k_idx * self.d_inner : (k_idx + 1) * self.d_inner, :]
                B_k = Bs_scan[:, k_idx, :, :]
                C_k = Cs_scan[:, k_idx, :, :]
                D_k = D_vec[k_idx * self.d_inner : (k_idx + 1) * self.d_inner]
                bias_k = bias_vec[k_idx * self.d_inner : (k_idx + 1) * self.d_inner]

                y_k = selective_scan_ref_pytorch(
                    u_k, dt_k, A_k, B_k, C_k, D_k, delta_bias=bias_k, delta_softplus=True
                )
                out_scans.append(y_k)
            out_y = torch.stack(out_scans, dim=1) # [B, K, d_inner, L]

        # 6. Hợp nhất 4 hướng quét và áp dụng cổng gating SiLU
        y_merged = self._merge_four_scans(out_y, H, W)
        y_merged = rearrange(y_merged, "b c h w -> b h w c")
        y_normed = self.out_norm(y_merged)

        y_gated = y_normed * F.silu(z_branch)
        out_hv = self.out_proj(y_gated)
        out_hv = rearrange(out_hv, "b h w c -> b c h w")

        return out_hv

# Alias để dễ import trong các file net
IG_Mamba = IlluminationGuidedDeltaMamba
