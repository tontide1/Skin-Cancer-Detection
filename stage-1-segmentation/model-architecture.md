# Kiến trúc U-Net với Backbone ResNet34 + Attention Decoder

Đây là bản mô tả chi tiết về kiến trúc mô hình phân đoạn tổn thương da (Skin Lesion Segmentation) sử dụng **U-Net** với **ResNet34** làm encoder và **attention ở decoder**.

Mô hình này là một mạng **Fully Convolutional Network (FCN)** có hình dạng chữ U, bao gồm 3 phần chính:
1. **Encoder (Contracting Path):** Trích xuất đặc trưng đa tầng
2. **Decoder (Expanding Path):** Khôi phục không gian và tái tạo mask
3. **Skip Connections:** Kết nối tắt giữa encoder và decoder

---

## 1. Tổng quan Kiến trúc

### 1.1. Thông số chính
* **Đầu vào:** Ảnh màu RGB, kích thước \(H \times W \times 3\) (Ví dụ: \(256 \times 256 \times 3\))
* **Đầu ra:** Mask phân đoạn nhị phân, kích thước \(H \times W \times 1\) (Giá trị logits cho mỗi pixel)
* **Số tầng encoder:** 5 stages (downsampling từ 1/2 đến 1/32 so với ảnh gốc)
* **Số tầng decoder:** 5 blocks (upsampling từ 1/32 về 1/1)
* **Encoder backbone:** ResNet34 (BasicBlock) với pretrained weights từ ImageNet

---

## 2. Encoder: ResNet34 Backbone

Encoder sử dụng **ResNet34** – một biến thể ResNet dùng **BasicBlock** (2 lớp Conv 3×3) *không có grouped conv, không có SE*, nhẹ hơn nhưng vẫn đủ sâu cho bài toán segmentation với 3k–5k ảnh.

### 2.1. Stage 0: Stem Block (Khối khởi đầu)

**Mục đích:** Giảm nhanh kích thước không gian để tiết kiệm tính toán.

**Cấu trúc:**
```
Input (3 channels) 
  → Conv2d(7×7, stride=2, padding=3) 
  → BatchNorm2d 
  → ReLU
  → MaxPool2d(3×3, stride=2, padding=1)
```

**Output:**
- Kích thước: \(H/4 \times W/4\) (ví dụ: \(64 \times 64\))
- Số kênh: **64**
- Downsampling: **1/4**

---

### 2.2. Cấu trúc Residual BasicBlock (ResNet34)

Mỗi residual block trong ResNet34 có dạng **BasicBlock**:

```
Input (C_in channels)
  ↓
1. Conv2d(3×3, stride=s, padding=1)
  ↓
2. BatchNorm2d
  ↓
3. ReLU
  ↓
4. Conv2d(3×3, stride=1, padding=1)
  ↓
5. BatchNorm2d
  ↓
6. Add residual connection (identity hoặc Conv1×1 nếu thay đổi kênh/stride)
  ↓
7. ReLU
```

**Đặc điểm:**
- Kernel 3×3 ở cả 2 conv → rất phù hợp cho segmentation (nhận diện biên, texture).
- Không dùng grouped conv, không SE → **ít tham số, giảm nguy cơ overfitting** trên dataset vừa (3k–5k ảnh).
- Vẫn giữ được lợi thế **skip connection** trong từng block → gradient flow ổn định.

---

### 2.3. Các Stage trong Encoder (ResNet34)

| Stage | Số blocks | Input size | Output size | Output channels | Downsampling |
|:------|:----------|:-----------|:------------|:----------------|:-------------|
| **Stage 0** (Stem) | 1 | \(256 \times 256\) | \(64 \times 64\) | 64  | 1/4  |
| **Stage 1** (layer1) | 3 | \(64 \times 64\)  | \(64 \times 64\) | 64  | 1/4  |
| **Stage 2** (layer2) | 4 | \(64 \times 64\)  | \(32 \times 32\) | 128 | 1/8  |
| **Stage 3** (layer3) | 6 | \(32 \times 32\)  | \(16 \times 16\) | 256 | 1/16 |
| **Stage 4** (layer4) | 3 | \(16 \times 16\)  | \(8 \times 8\)   | 512 | 1/32 |

**Ghi chú:**
- **Stage 1–4** sử dụng BasicBlock như mô tả ở trên (expansion = 1).
- **Không có SE / grouped conv** trong encoder → mô hình **nhẹ hơn SE-ResNeXt50**, dễ train hơn trên dataset vừa phải.
- **Stage 4** vẫn là bottleneck của U-Net (feature maps kích thước nhỏ nhất, chứa nhiều thông tin ngữ nghĩa nhất).

---

## 3. Decoder: Expanding Path + Attention

Decoder có nhiệm vụ **khôi phục không gian** (spatial resolution) từ feature maps trừu tượng (từ bottleneck) về kích thước gốc, đồng thời **kết hợp thông tin chi tiết** từ encoder thông qua skip connections **với attention**.

### 3.1. Cấu trúc của một Decoder Block (có Attention)

Mỗi decoder block gồm 4 bước:

```
Input: Feature maps từ tầng sâu hơn (decoder) + feature maps từ encoder (skip)
  ↓
1. UPSAMPLING (×2):
   - Phóng to feature maps decoder lên gấp 2 (H×W → 2H×2W)
   - Phương pháp: Bilinear + Conv
  ↓
2. ATTENTION GATE (trên skip connection):
   - Nhận:
       + g: feature từ decoder (đã upsample)
       + x: feature từ encoder (skip)
   - Tính:
       + α = σ(Conv1×1(ReLU(Conv1×1(g) + Conv1×1(x))))
       + x_att = α ⊙ x   (⊙: nhân theo từng phần tử)
   - Kết quả: skip features đã được “làm sạch”, tập trung vào vùng tổn thương
  ↓
3. CONCATENATION:
   - Ghép nối upsampled decoder features với x_att
   - Operation: torch.cat([g, x_att], dim=1)
  ↓
4. CONVOLUTION BLOCKS:
   - Conv2d(3×3) + BatchNorm + ReLU
   - Conv2d(3×3) + BatchNorm + ReLU
   - Mục đích: Hòa trộn thông tin từ 2 nguồn (deep features + attended skip features)
  ↓
Output: Refined feature maps
```

**Ý nghĩa của từng bước:**
- **Upsampling:** Tăng độ phân giải không gian.
- **Attention gate:** Loại bớt background không liên quan trên skip (vùng da lành, lông, thước đo, noise).
- **Skip connections:** Bổ sung thông tin chi tiết (edges, texture) từ encoder **nhưng đã qua attention**.
- **Convolution:** Học cách kết hợp 2 loại thông tin (semantic + spatial details).

#### 3.1.1. Attention Gate – Cách triển khai (Pseudo-code)

**Ký hiệu:**
- \(x\): feature từ encoder (skip), shape \((B, C_\text{enc}, H, W)\)
- \(g\): feature từ decoder (gating), shape \((B, C_\text{dec}, H, W)\) – sau khi đã được upsample về cùng \(H, W\)
- \(C_\text{int}\): số kênh trung gian trong attention gate (thường chọn nhỏ hơn \(C_\text{enc}, C_\text{dec}\))

**Công thức:**
\[
\begin{aligned}
q_x &= W_x * x \quad &&\text{(Conv 1×1 giảm kênh encoder)} \\
q_g &= W_g * g \quad &&\text{(Conv 1×1 giảm kênh decoder)} \\
f &= \text{ReLU}(q_x + q_g) \\
\psi &= \sigma(W_\psi * f) \quad &&\text{(Conv 1×1 rồi Sigmoid, }\psi \in [0,1]^{B\times 1\times H\times W}) \\
x_\text{att} &= x \odot \psi \quad &&\text{(nhân theo từng phần tử, broadcast theo kênh)}
\end{aligned}
\]

**Pseudo-code kiểu PyTorch:**

```python
import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    def __init__(self, in_channels_encoder: int, in_channels_decoder: int, inter_channels: int):
        super().__init__()
        # Giảm kênh của skip (encoder) và gating (decoder) về cùng inter_channels
        self.theta_x = nn.Conv2d(in_channels_encoder, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.phi_g = nn.Conv2d(in_channels_decoder, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Tạo map attention (1 channel) rồi sigmoid
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        x: skip features từ encoder  (B, C_enc, H, W)
        g: gating từ decoder (upsampled) (B, C_dec, H, W)
        """
        # Đưa về cùng số kênh trung gian
        theta_x = self.theta_x(x)   # (B, C_int, H, W)
        phi_g = self.phi_g(g)       # (B, C_int, H, W)

        # Kết hợp và kích hoạt
        f = self.relu(theta_x + phi_g)

        # Map attention (1 channel) + sigmoid
        psi = self.sigmoid(self.psi(f))  # (B, 1, H, W), giá trị ∈ [0, 1]

        # Nhân với skip gốc (broadcast theo kênh)
        x_att = x * psi                  # (B, C_enc, H, W)
        return x_att
```

**Gợi ý cấu hình các block:**
- **Decoder 4:**  
  - \(x\): từ Stage 3, \(C_\text{enc} = 256\)  
  - \(g\): từ Stage 4 (upsample), \(C_\text{dec} = 512\) hoặc 256 (sau Conv)  
  - \(C_\text{int}\): 128
- **Decoder 3:** \(C_\text{enc} = 128, C_\text{dec} = 256, C_\text{int} = 64\)
- **Decoder 2:** \(C_\text{enc} = 64,  C_\text{dec} = 128, C_\text{int} = 32\)
- **Decoder 1:** \(C_\text{enc} = 64,  C_\text{dec} = 64,  C_\text{int} = 32\)

Các giá trị trên có thể điều chỉnh tùy theo số kênh thực tế bạn thiết kế trong decoder, nhưng nguyên tắc chung là:
- \(C_\text{int}\) nhỏ hơn \(C_\text{enc}, C_\text{dec}\) để giảm chi phí tính toán.
- Kích thước không gian \(H, W\) của \(x\) và \(g\) phải **bằng nhau** trước khi đưa vào attention gate.

---

### 3.2. Chi tiết các Decoder Blocks

| Block | Input từ | Skip từ | Input size | Skip size | Output size | Output channels |
|:------|:---------|:--------|:-----------|:----------|:------------|:----------------|
| **Decoder 4** | Stage 4 | Stage 3 | \(8 \times 8\) | \(16 \times 16\) | \(16 \times 16\) | 256 |
| **Decoder 3** | Decoder 4 | Stage 2 | \(16 \times 16\) | \(32 \times 32\) | \(32 \times 32\) | 128 |
| **Decoder 2** | Decoder 3 | Stage 1 | \(32 \times 32\) | \(64 \times 64\) | \(64 \times 64\) | 64 |
| **Decoder 1** | Decoder 2 | Stage 0 | \(64 \times 64\) | \(64 \times 64\) | \(128 \times 128\) | 32 |
| **Decoder 0** | Decoder 1 | (none) | \(128 \times 128\) | - | \(256 \times 256\) | 16 |

**Ghi chú quan trọng:**
- **Decoder 4–1:** Có skip connections từ encoder tương ứng **và áp dụng attention gate** trước khi concatenate.
- **Decoder 0:** Không có skip connection, chỉ upsample và convolution.
- **Số channels giảm dần:** 256 → 128 → 64 → 32 → 16 (giảm độ phức tạp khi tiến về output).

---

### 3.3. Vai trò của Skip Connections + Attention

Skip connections là **cầu nối** giữa encoder và decoder, còn attention gate giúp **lọc chọn** những thông tin quan trọng trước khi truyền qua:

**1. Bảo toàn thông tin chi tiết (Fine-grained details)**
- Encoder stages gần input chứa thông tin spatial chi tiết (edges, texture).
- Attention gate giữ lại vùng liên quan đến tổn thương, giảm background.
- Decoder sử dụng thông tin này để vẽ biên giới chính xác.

**2. Giải quyết vấn đề mất thông tin (Information loss)**
- Các lớp downsampling trong encoder → mất thông tin không gian.
- Skip connections → bù đắp thông tin bị mất.
- Attention → tránh đưa quá nhiều noise từ encoder xuống decoder.

**3. Cải thiện gradient flow**
- Tạo đường đi ngắn hơn cho gradient → training ổn định hơn.
- Attention giúp gradient tập trung vào vùng lesion nhiều hơn.

**Ví dụ cụ thể:**
- **Stage 0 skip → Decoder 1:** Cung cấp thông tin cạnh sắc nét của tổn thương (sau khi đã suppress vùng da lành).
- **Stage 3 skip → Decoder 4:** Cung cấp semantic context (vùng tổn thương nằm ở đâu) với attention tập trung vào vùng cần phân đoạn.

---

## 4. Segmentation Head (Đầu ra phân đoạn)

Lớp cuối cùng của mô hình, chuyển đổi feature maps thành mask phân đoạn.

### 4.1. Cấu trúc

```
Input: Decoder output (256×256×16)
  ↓
Conv2d(1×1, in_channels=16, out_channels=1)
  ↓
Output: Logits map (256×256×1)
```

### 4.2. Đặc điểm

**1. Pointwise Convolution (1×1)**
- **Mục đích:** Chiếu 16 feature channels xuống 1 channel duy nhất
- **Không có activation function:** Output là raw logits (giá trị thực không giới hạn)

**2. Output Format**
- **Shape:** \((B, 1, H, W)\) - Batch size × 1 channel × Height × Width
- **Giá trị:** Logits ∈ ℝ (có thể âm hoặc dương)
- **Ý nghĩa:** 
  - Logit > 0 → Có khả năng là tổn thương cao
  - Logit < 0 → Có khả năng là background cao

**3. Lý do không dùng Sigmoid trong model**
- **Loss function:** Sử dụng `BCEWithLogitsLoss` (đã tích hợp Sigmoid bên trong)
- **Ưu điểm:** 
  - Numerical stability (tránh log(0) errors)
  - Hiệu quả tính toán (kết hợp Sigmoid + BCE trong 1 operation)

**4. Inference (Khi dự đoán)**
```python
# Training: model trả về logits
logits = model(x)  # Shape: (B, 1, H, W)

# Inference: cần áp dụng Sigmoid để có xác suất
probs = torch.sigmoid(logits)  # Shape: (B, 1, H, W), values ∈ [0, 1]

# Threshold để có binary mask
mask = (probs > 0.5).float()  # Shape: (B, 1, H, W), values ∈ {0, 1}
```

---

## 5. Tóm tắt Luồng Dữ liệu (Data Flow)

Bảng dưới đây tóm tắt luồng dữ liệu qua toàn bộ mạng (giả sử input \(256 \times 256\)):

| Thành phần | Tên | Kích thước | Channels | Skip Connection | Ghi chú |
|:-----------|:----|:-----------|:---------|:----------------|:--------|
| **INPUT** | Input Image | \(256 \times 256\) | 3 | - | RGB image |
| | | | | | |
| **ENCODER** | | | | | |
| ↓ | Stage 0 (Stem)   | \(64 \times 64\)  | 64  | → Decoder 1 | Conv7×7 + MaxPool |
| ↓ | Stage 1 (layer1) | \(64 \times 64\)  | 64  | → Decoder 2 | 3 BasicBlocks |
| ↓ | Stage 2 (layer2) | \(32 \times 32\)  | 128 | → Decoder 3 | 4 BasicBlocks |
| ↓ | Stage 3 (layer3) | \(16 \times 16\)  | 256 | → Decoder 4 | 6 BasicBlocks |
| ↓ | **Stage 4 (layer4)** | **\(8 \times 8\)** | **512** | - | **3 BasicBlocks – Bottleneck** |
| **DECODER** | | | | | |
| ↑ | Decoder Block 4 | \(16 \times 16\) | 256 | ← Stage 3 | Upsample + Attention + Skip + Conv |
| ↑ | Decoder Block 3 | \(32 \times 32\) | 128 | ← Stage 2 | Upsample + Attention + Skip + Conv |
| ↑ | Decoder Block 2 | \(64 \times 64\) | 64  | ← Stage 1 | Upsample + Attention + Skip + Conv |
| ↑ | Decoder Block 1 | \(128 \times 128\) | 32 | ← Stage 0 | Upsample + Attention + Skip + Conv |
| ↑ | Decoder Block 0 | \(256 \times 256\) | 16 | - | Upsample + Conv |
| | | | | | |
| **OUTPUT** | Segmentation Head | \(256 \times 256\) | 1 | - | Conv 1×1 → Logits |

---

## 6. Điểm mạnh của Kiến trúc

### 6.1. Khả năng biểu diễn mạnh mẽ nhưng nhẹ

**1. ResNet34 Backbone**
- **512 channels ở bottleneck:** Đủ capacity để học semantic features phức tạp cho bài toán 3k–5k ảnh.
- **BasicBlock với Conv 3×3:** Rất hợp với segmentation (nhạy với edges, texture).
- **Số tham số vừa phải:** Nhẹ hơn SE-ResNeXt50 → giảm overfitting, train nhanh hơn trên 2×T4.
- **Pretrained trên ImageNet:** Transfer learning giúp tăng tốc hội tụ và cải thiện performance.

**2. Attention trong Decoder**
- **Attention gate trên từng skip connection:** Tự động tập trung vào vùng lesion, giảm background noise.
- **Channel + spatial gating (gián tiếp qua conv 1×1 + Sigmoid):** Học được vùng “nên mở” và “nên tắt” trên feature maps.
- **Lọc nhiễu hiệu quả:** Giảm ảnh hưởng của artifacts (lông, thước đo, bong bóng) ngay tại decoder.
- **Tăng cường signal tại biên tổn thương:** Giúp mask sắc nét, đặc biệt cho lesion nhỏ hoặc biên mỏng.

### 6.2. Bảo toàn chi tiết không gian

**Skip Connections (U-Net architecture)**
- **Multi-scale information fusion:** Kết hợp features từ nhiều độ phân giải khác nhau
- **Precise boundary localization:** Giữ được thông tin edges và boundaries chi tiết
- **Gradient flow:** Giúp training stable với mạng rất sâu (50+ layers)

### 6.3. Phù hợp với bài toán phân đoạn tổn thương da

**1. Xử lý texture phức tạp**
- Tổn thương da có texture đa dạng (sần sùi, vảy, nhám)
- Backbone CNN nhiều tầng (ResNet34) → học được nhiều texture patterns

**2. Phân biệt tổn thương vs artifacts**
- Attention gate ở decoder → giảm ảnh hưởng của lông, thước đo, bong bóng.
- Multi-scale features → hiểu context xung quanh.

**3. Boundaries chính xác**
- Skip connections → giữ được thông tin edges từ early layers
- Critical cho medical imaging (cần segmentation chính xác)

---

## 7. Thông số Mô hình

### Tổng quan
- **Tổng số parameters:** ≈ **23M–24M parameters** (ResNet34 encoder + decoder + attention)
- **Input size:** 256×256
- **Output size:** Same as input (pixel-wise segmentation)

### Khuyến nghị sử dụng
- **Training:** Mixed precision (FP16) để tiết kiệm memory
- **Data augmentation:** Essential (rotations, flips, color jitter, elastic transforms)
- **Loss function:** Combined loss (BCE + Dice)
- **Optimizer:** AdamW với learning rate scheduling
- **Batch size:** 8-16 (tùy GPU memory)

---

## 8. Loss Function: BCEWithLogitsLoss + DiceLoss

### 8.1. Lý do chọn 0.5 × BCE + 0.5 × Dice

- **BCEWithLogitsLoss:**
  - Hoạt động trên **từng pixel độc lập** (per-pixel classification).
  - Phù hợp cho việc tối ưu xác suất / calibration (logits → xác suất).
  - Nhưng có thể **bị lệch** khi class imbalance lớn (lesion rất nhỏ so với background).

- **DiceLoss:**
  - Đo trực tiếp **mức độ chồng lấp giữa vùng dự đoán và vùng ground truth** (region-level overlap).
  - Rất **nhạy với class imbalance**, vì tính trên tổng vùng, không phải từng pixel riêng lẻ.
  - Tối ưu trực tiếp mục tiêu giống các metric như Dice / IoU.

- **Kết hợp 0.5 / 0.5:**
  - **BCE** giúp mô hình học tín hiệu local, ổn định gradient cho từng pixel.
  - **Dice** buộc mô hình tối ưu theo hình dạng vùng (shape) và kích thước lesion.
  - Hệ số 0.5–0.5 là một **điểm khởi đầu cân bằng**, phù hợp cho đa số bài toán binary segmentation, đặc biệt với Attention U-Net.

Trong thực tế bạn có thể tinh chỉnh:
- Nếu lesion **rất nhỏ, class imbalance nặng** → tăng trọng số Dice:  
  `loss = 0.3 * BCE + 0.7 * Dice`
- Nếu mô hình khó hội tụ, logits rất nhiễu → tăng trọng số BCE một chút:  
  `loss = 0.7 * BCE + 0.3 * Dice`

### 8.2. Định nghĩa Dice Loss (Soft Dice)

Giả sử:
- \(p = \sigma(\text{logits})\): xác suất dự đoán sau Sigmoid, \(p \in [0,1]\)
- \(y\): ground truth mask, \(y \in \{0,1\}\)

Soft Dice cho mỗi mẫu trong batch:
\[
\text{Dice}(p, y) = \frac{2 \sum (p \cdot y) + \epsilon}{\sum p + \sum y + \epsilon}
\]

Trong đó \(\epsilon\) là số nhỏ để tránh chia cho 0.  
Dice Loss:
\[
\text{DiceLoss} = 1 - \text{Dice}(p, y)
\]

### 8.3. Pseudo-code cho Combined Loss

```python
import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    logits: (B, 1, H, W) - đầu ra thô từ model
    targets: (B, 1, H, W) - mask nhị phân {0, 1}
    """
    probs = torch.sigmoid(logits)          # (B, 1, H, W), ∈ [0, 1]
    targets = targets.float()

    # Tính Dice theo batch
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice.mean()
    return loss


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dsc
```

**Yêu cầu để logic loss đúng với kiến trúc:**
- Model **trả về logits** (không Sigmoid trong `forward`) → phù hợp với `BCEWithLogitsLoss`.
- `dice_loss` phải **tự Sigmoid bên trong**, không Sigmoid hai lần.
- `target` có shape giống `pred` (B, 1, H, W) và giá trị {0, 1}.
- Khi log metric (Dice / IoU) trong validation, nên tính lại Dice trên **probs = sigmoid(logits)** với ngưỡng (thường 0.5).

---

## 9. Pretrained Loading với `segmentation_models_pytorch`

### 9.1. Cấu hình model trong thực nghiệm

Nếu bạn sử dụng thư viện `segmentation_models_pytorch` (SMP), một cấu hình thực tế tương ứng với kiến trúc đã mô tả là:

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    decoder_attention_type="scse",  # attention ở decoder (squeeze & excitation + spatial)
    in_channels=3,
    classes=1,
)
```

**Giải thích các tham số:**
- **`encoder_name="resnet34"`**:  
  - Dùng backbone ResNet34 pretrained ImageNet đúng như phần Encoder đã mô tả.
- **`encoder_weights="imagenet"`**:  
  - Load trọng số pretrained trên ImageNet cho encoder → tốt cho transfer learning trên dataset 3k–5k ảnh.
- **`decoder_attention_type="scse"`**:  
  - Kích hoạt **SCSE (Spatial and Channel Squeeze & Excitation)** ở từng block decoder.  
  - Đây là attention dạng re-weighting channel + spatial, khác một chút so với attention gate trên skip, nhưng cùng ý tưởng: nhấn mạnh vùng quan trọng, giảm nhiễu.
- **`in_channels=3`**:  
  - Ảnh RGB đầu vào.
- **`classes=1`**:  
  - Phân đoạn nhị phân (lesion vs background), output 1 kênh logits như phần Segmentation Head.

### 9.2. Gắn với `combined_loss`

Trong vòng lặp huấn luyện, bạn có thể sử dụng:

```python
for images, masks in train_loader:
    images = images.to(device)          # (B, 3, H, W)
    masks = masks.to(device)           # (B, 1, H, W), {0, 1}

    logits = model(images)             # (B, 1, H, W), raw logits
    loss = combined_loss(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Như vậy, phần **Pretrained Loading**, **Attention ở decoder** và **Loss function** đều nhất quán với nhau:
- Encoder ResNet34 pretrained.
- Decoder có attention (SCSE trong SMP, hoặc AttentionGate custom như phần 3 nếu bạn tự cài đặt).
- Loss là `0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss` hoạt động trực tiếp trên logits của model. 