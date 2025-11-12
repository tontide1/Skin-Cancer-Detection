### Kiến trúc đề xuất (chuẩn, nhất quán và thực dụng cho 256×256 dermoscopy)
- **Tổng quan**: ResNet‑34 encoder (ImageNet) → PPM bottleneck → U‑Net decoder 5 tầng (dùng cả skip nông ở 128×128) → Attention Gate trên skip + scSE sau hợp nhất → đầu ra 1 kênh, sigmoid khi suy luận.  
- **Upsample**: bilinear (align_corners=False) + Conv 3×3 (tránh checkerboard).  
- **Chuẩn hóa**: BN cố định ở encoder; decoder dùng GN hoặc BN (tùy batch size).  

#### Encoder (ResNet‑34)
- Input: 256×256×3
- Stem: Conv7×7 s=2 → 128×128×64 = `E0`
- MaxPool s=2 → 64×64×64
- Layer1 → 64×64×64 = `E1`
- Layer2 → 32×32×128 = `E2`
- Layer3 → 16×16×256 = `E3`
- Layer4 → 8×8×512 = `E4`

Skip sẽ dùng: `E0` (128×128×64), `E1` (64×64×64), `E2` (32×32×128), `E3` (16×16×256)

#### Bottleneck (PPM để tăng ngữ cảnh)
- Input: 8×8×512 (`E4`)
- Nhánh PPM: AdaptiveAvgPool2d(output_size ∈ {1,2,3,6}) → Conv1×1 giảm kênh (mỗi nhánh ~128) → upsample về 8×8
- Nối [E4, các nhánh] → 8×8×(512+4×128)=8×8×1024 → Conv3×3 → BN/GN → ReLU → Dropout(0.2–0.3) → 8×8×512 = `B`

#### Decoder (5 tầng với Attention Gate ở skip, scSE sau concat)
**Lưu ý**: Tất cả upsample dùng `F.interpolate(mode='bilinear', align_corners=False)` để tránh grid misalignment.

- Khối D4 (8→16):
  - Up(B): Upsample bilinear ×2 (align_corners=False) → Conv3×3: 8×8×512 → 16×16×256
  - Attention Gate trên skip `E3` (16×16×256) dùng tín hiệu từ nhánh decoder 16×16×256 → `E3_att` (16×16×256)
  - Concat [upsampled, `E3_att`] → 16×16×512
  - Conv3×3 → BN/GN → ReLU → Conv3×3 → BN/GN → ReLU → scSE(r=16) → 16×16×256 = `D4`

- Khối D3 (16→32):
  - Up(`D4`): Upsample bilinear ×2 (align_corners=False) → Conv3×3: 16×16×256 → 32×32×128
  - AG trên `E2` (32×32×128) → `E2_att` (32×32×128)
  - Concat [upsampled, `E2_att`] → 32×32×256
  - Conv3×3 → BN/GN → ReLU → Conv3×3 → BN/GN → ReLU → scSE(r=16) → 32×32×128 = `D3`

- Khối D2 (32→64):
  - Up(`D3`): Upsample bilinear ×2 (align_corners=False) → Conv3×3: 32×32×128 → 64×64×64
  - AG trên `E1` (64×64×64) → `E1_att` (64×64×64)
  - Concat [upsampled, `E1_att`] → 64×64×128
  - Conv3×3 → BN/GN → ReLU → Conv3×3 → BN/GN → ReLU → scSE(r=8) → 64×64×64 = `D2`

- Khối D1 (64→128, dùng skip nông E0):
  - Up(`D2`): Upsample bilinear ×2 (align_corners=False) → Conv3×3: 64×64×64 → 128×128×32
  - Skip E0 (128×128×64) → Conv1×1(64→32) → 128×128×32 (projection bắt buộc để khớp kênh)
  - AG trên skip đã project (128×128×32) dùng tín hiệu từ decoder 128×128×32 → `E0_att` (128×128×32)
  - Concat [upsampled, `E0_att`] → 128×128×64
  - Conv3×3 → BN/GN → ReLU → Conv3×3 → BN/GN → ReLU → scSE(r=8) → 128×128×32 = `D1`

- Đầu ra (Final Upsampling):
  - Up(`D1`): Upsample bilinear ×2 (align_corners=False): 128×128×32 → 256×256×32
  - Conv3×3 tinh chỉnh: 256×256×32 → 256×256×32
  - Conv1×1: 32→1 → logits 256×256×1
  - Sigmoid activation (chỉ khi inference, training dùng logits cho BCEWithLogitsLoss)

### Attention “đúng” và tối ưu (định nghĩa ngắn gọn)
- **Attention Gate (AG) trên skip** (theo Attention U‑Net):
  - Với skip `x` (H×W×C_s) và gating `g` (từ nhánh decoder, H×W×C_g):
    - `q = ReLU(Conv1×1(x) + Conv1×1(g))`
    - `α = Sigmoid(Conv1×1(q))`  (H×W×1)
    - `x_att = x ⊙ α`  (nhân theo không gian, giữ kênh)
  - Mục tiêu: lọc skip bằng ngữ cảnh sâu từ decoder, giảm nhiễu nền.

- **scSE sau hợp nhất**:
  - cSE: GAP → Conv1×1(C→C/r) → ReLU → Conv1×1(C/r→C) → Sigmoid → nhân theo kênh.
  - sSE: Conv1×1(C→1) → Sigmoid → nhân theo không gian.
  - Fusion: `y = cSE(x) + sSE(x)` (r=16; với C≤64 có thể r=8).
