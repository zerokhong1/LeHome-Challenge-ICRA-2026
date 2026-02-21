# LeHome-Challenge-ICRA-2026

# Phân Tích Toán Học: ACT (CVAE) & Diffusion Policy (DDPM)
## Dành cho LeHome Challenge — ICRA 2026

> **Prerequisite**: Biết probability basics (Bayes' rule, KL divergence, Gaussian), neural networks, gradient descent.  
> **Notation convention**: Bold lowercase = vectors, bold uppercase = matrices, calligraphic = distributions/spaces.

---

## PHẦN I: BEHAVIOR CLONING — ĐIỂM XUẤT PHÁT

### 1.1 Formulation

Imitation learning problem: cho dataset expert demonstrations

$$\mathcal{D} = \{(\mathbf{o}_1, \mathbf{a}_1), (\mathbf{o}_2, \mathbf{a}_2), \ldots, (\mathbf{o}_N, \mathbf{a}_N)\}$$

trong đó:
- **o_t** ∈ ℝ^d_o : observation tại timestep t (images + joint positions)
- **a_t** ∈ ℝ^d_a : action tại timestep t (target joint angles)

Mục tiêu: học policy π_θ(**a** | **o**) sao cho minimize kỳ vọng sai lệch so với expert.

**Behavior Cloning** đơn giản là supervised learning:

$$\theta^* = \arg\min_\theta \mathbb{E}_{(\mathbf{o}, \mathbf{a}) \sim \mathcal{D}} \left[ \mathcal{L}(\pi_\theta(\mathbf{o}), \mathbf{a}) \right]$$

Nếu dùng MSE loss (deterministic policy):

$$\mathcal{L} = \|\pi_\theta(\mathbf{o}) - \mathbf{a}\|_2^2$$

### 1.2 Vấn đề #1: Compounding Error

Gọi ε = probability policy sai tại mỗi step, T = horizon length.

**Bound của BC:**

$$J(\pi_\theta) \leq J(\pi^*) + \mathcal{O}(\varepsilon T^2)$$

Tại sao T²? Vì tại step t, xác suất đã drift khỏi expert distribution ~ εt (linear accumulation), nhưng mỗi drift tạo cost ~ εt nữa (vì state càng xa expert → cost càng lớn). Tổng: Σ_{t=1}^{T} εt ≈ εT²/2.

**So sánh**: DAgger đạt O(εT) vì iteratively correct distribution shift.

### 1.3 Vấn đề #2: Multimodal Actions

Xét task gấp áo: tại cùng một observation, expert A gấp từ trái sang phải, expert B gấp từ phải sang trái. Cả hai đều đúng.

Nếu dùng MSE loss, policy sẽ predict **trung bình** của hai actions → gấp ở giữa → sai hoàn toàn!

$$\mathbf{a}_{avg} = \frac{1}{2}(\mathbf{a}_{left} + \mathbf{a}_{right}) \neq \text{any valid action}$$

Đây gọi là **mode averaging** — vấn đề cốt lõi mà cả ACT và Diffusion Policy phải giải quyết.

---

## PHẦN II: ACT — CONDITIONAL VAE FORMULATION

### 2.1 Tại sao CVAE?

ACT sử dụng Conditional Variational Autoencoder (CVAE) để giải quyết multimodality. Ý tưởng: thêm một latent variable **z** đại diện cho "style" của action sequence, cho phép cùng một observation map sang nhiều action modes khác nhau.

**Kết nối với 3DGS**: Trong 3D Gaussian Splatting, mỗi Gaussian có mean (position), covariance (shape), và opacity. Tương tự, latent z trong CVAE encode "dạng" của trajectory — cùng một scene nhưng z khác nhau cho trajectories khác nhau.

### 2.2 CVAE Background

#### 2.2.1 VAE Recap

Standard VAE cho data **x**:
- **Encoder** (recognition model): q_φ(**z** | **x**) — approximate posterior
- **Decoder** (generative model): p_θ(**x** | **z**) — likelihood
- **Prior**: p(**z**) = N(0, I)

ELBO (Evidence Lower Bound):

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x} | \mathbf{z})] - D_{KL}(q_\phi(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$$

Maximize ELBO = maximize log-likelihood lower bound. Hai member:
1. **Reconstruction term**: decoder phải tái tạo tốt x từ z
2. **KL term**: encoder phải gần prior → regularization, ép z có cấu trúc

#### 2.2.2 Conditional VAE (CVAE)

CVAE thêm conditioning variable **c** (trong ACT: **c** = observation):

- **Encoder**: q_φ(**z** | **x**, **c**) — giờ condition on cả data và observation
- **Decoder**: p_θ(**x** | **z**, **c**) — generate data given z và observation
- **Prior**: p_ψ(**z** | **c**) — prior cũng có thể condition on **c** (hoặc giữ p(**z**) = N(0,I) cho đơn giản)

CVAE ELBO:

$$\log p_\theta(\mathbf{x} | \mathbf{c}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c})} [\log p_\theta(\mathbf{x} | \mathbf{z}, \mathbf{c})] - D_{KL}(q_\phi(\mathbf{z} | \mathbf{x}, \mathbf{c}) \| p(\mathbf{z}))$$

### 2.3 ACT Architecture — Chi tiết toán học

Trong ACT, mapping cụ thể:

| CVAE concept | ACT mapping |
|---|---|
| **x** (data) | **a**_{t:t+k} = action chunk (k future actions) |
| **c** (condition) | **o**_t = (images, joint positions) |
| **z** (latent) | "style" variable ∈ ℝ^d_z |
| Encoder q_φ | Transformer encoder trên (action_seq + joint_pos) |
| Decoder p_θ | Transformer encoder-decoder trên (images + joint_pos + z) |

#### 2.3.1 CVAE Encoder (chỉ dùng khi training)

Input:
- Action sequence ground truth: **a**_{t:t+k} ∈ ℝ^{k × d_a}
- Joint positions: **j**_t ∈ ℝ^{d_j}

Process:
1. Project actions qua linear layer: **h**_a^i = W_a · **a**_{t+i} + **b**_a, cho i = 0,...,k-1
2. Project joint positions: **h**_j = W_j · **j**_t + **b**_j
3. Thêm [CLS] token: **h**_cls ∈ ℝ^d (learnable)
4. Concat thành sequence: S_enc = [**h**_cls, **h**_j, **h**_a^0, ..., **h**_a^{k-1}]
5. Pass qua Transformer Encoder (self-attention):

$$\text{TransEnc}(S_{enc}) \to [\tilde{\mathbf{h}}_{cls}, \ldots]$$

6. Lấy output tại [CLS] position, project ra μ và log(σ²):

$$\boldsymbol{\mu} = W_\mu \tilde{\mathbf{h}}_{cls} + \mathbf{b}_\mu \in \mathbb{R}^{d_z}$$
$$\log \boldsymbol{\sigma}^2 = W_\sigma \tilde{\mathbf{h}}_{cls} + \mathbf{b}_\sigma \in \mathbb{R}^{d_z}$$

7. **Reparameterization trick** (giống standard VAE):

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Trick này cho phép backpropagate qua sampling operation (vì ε không phụ thuộc θ).

**Kết nối SLAM**: Reparameterization trick tương tự cách bạn perturb camera poses trong optimization — thay vì sample trực tiếp trên SE(3), bạn sample perturbation trong tangent space rồi exponential map lên manifold.

#### 2.3.2 CVAE Decoder (policy network — dùng cả training và inference)

Input:
- Multi-view images: {I_1, ..., I_C} (C cameras)
- Joint positions: **j**_t
- Latent variable: **z** (từ encoder khi training, = **0** khi inference)

Process:
1. **Visual encoding**: Mỗi image qua CNN backbone (ResNet-18 mặc định):

$$\mathbf{F}_c = \text{ResNet}(I_c) \in \mathbb{R}^{H' \times W' \times d_v}$$

   Flatten spatial dimensions → sequence of visual tokens:
   
$$[\mathbf{v}_c^1, \ldots, \mathbf{v}_c^{H'W'}] = \text{Flatten}(\mathbf{F}_c)$$

2. **Positional encoding**: Thêm 2D sinusoidal positional encoding cho spatial tokens (tương tự ViT).

3. **Condition tokens**: Concat tất cả visual tokens từ mọi cameras + projected joint positions + projected z:

$$S_{cond} = [\mathbf{v}_1^1, \ldots, \mathbf{v}_C^{H'W'}, \text{Proj}(\mathbf{j}_t), \text{Proj}(\mathbf{z})]$$

   Pass qua **Transformer Encoder** (self-attention trên conditions):

$$\mathbf{M} = \text{TransEnc}(S_{cond}) \quad \text{(memory for decoder)}$$

4. **Action query tokens**: k learnable query tokens Q = [**q**_0, ..., **q**_{k-1}], mỗi cái đại diện cho một timestep trong action chunk.

5. **Transformer Decoder** với cross-attention:

$$[\hat{\mathbf{h}}_0, \ldots, \hat{\mathbf{h}}_{k-1}] = \text{TransDec}(Q, \mathbf{M})$$

   Trong đó:
   - Self-attention giữa các query tokens (cho temporal coherence)
   - Cross-attention: query tokens attend vào memory M (visual + proprioceptive + latent info)

6. **Action prediction**: Linear projection cho mỗi timestep:

$$\hat{\mathbf{a}}_{t+i} = W_{out} \hat{\mathbf{h}}_i + \mathbf{b}_{out} \in \mathbb{R}^{d_a}, \quad i = 0, \ldots, k-1$$

#### 2.3.3 Loss Function

$$\mathcal{L}_{ACT} = \underbrace{\sum_{i=0}^{k-1} \|\mathbf{a}_{t+i} - \hat{\mathbf{a}}_{t+i}\|_1}_{\text{L1 Reconstruction Loss}} + \underbrace{\beta \cdot D_{KL}(q_\phi(\mathbf{z} | \mathbf{a}, \mathbf{o}) \| \mathcal{N}(\mathbf{0}, \mathbf{I}))}_{\text{KL Regularization}}$$

**Về L1 loss** (không phải L2): L1 loss robust hơn với outliers trong demonstrations. Khi expert demonstrations có noise (tay người rung), L1 không penalize large deviations as heavily as L2.

**Về KL term** với weight β:

$$D_{KL}(q_\phi(\mathbf{z}|\mathbf{a}, \mathbf{o}) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) = \frac{1}{2} \sum_{j=1}^{d_z} \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

(Closed-form vì cả hai đều Gaussian)

**β tuning rất quan trọng**:
- β quá lớn → "posterior collapse": z bị ép về prior, decoder ignore z → mất khả năng model multimodality
- β quá nhỏ → z encode quá nhiều info, decoder "cheat" bằng z thay vì dùng visual observations → generalization kém
- Paper dùng β = 10 (khá nhỏ so với standard VAE, ưu tiên reconstruction)

#### 2.3.4 Inference Time

```
1. z = 0  (mean of prior — no sampling!)
2. Encode observations → memory M
3. Decode with query tokens → action chunk [â_{t}, ..., â_{t+k-1}]
4. Execute first few actions, then re-query
```

**Tại sao z = 0 works?** Vì KL regularization ép posterior q(z|a,o) gần N(0,I). Khi inference, set z = 0 (mode of prior) → decoder nhận z phổ biến nhất → generate "average best" action cho mode chính. Multimodality bị collapse về 1 mode, nhưng mode đó thường là mode tốt nhất.

### 2.4 Temporal Ensembling

Thay vì execute toàn bộ chunk rồi query lại, ACT query policy mỗi timestep nhưng dùng exponential weighting để combine overlapping predictions:

Tại timestep t, có thể có nhiều predictions cho action a_t:
- Prediction từ query tại t: chunk[0]
- Prediction từ query tại t-1: chunk[1]
- ...
- Prediction từ query tại t-m: chunk[m] (nếu m < k)

Temporal ensembling:

$$\mathbf{a}_t^{final} = \frac{\sum_{m=0}^{\min(t, k-1)} w_m \cdot \hat{\mathbf{a}}_t^{(t-m)}}{\sum_{m=0}^{\min(t, k-1)} w_m}$$

trong đó w_m = exp(-λ·m), λ > 0 là decay rate. Predictions gần đây (m nhỏ) được weight cao hơn.

**Tại sao hiệu quả?**
- Smoothing: trung bình giảm noise/jitter
- Implicit closed-loop: predictions mới (dựa trên observations mới) được ưu tiên
- Error correction: nếu chunk cũ sai, chunks mới (từ state mới) có thể correct

---

## PHẦN III: DIFFUSION POLICY — DDPM FORMULATION

### 3.1 Denoising Diffusion Probabilistic Models (DDPM) — Background

#### 3.1.1 Forward Process (thêm noise)

Cho clean data **x**_0 ~ q(**x**_0), forward process thêm Gaussian noise qua K steps:

$$q(\mathbf{x}_k | \mathbf{x}_{k-1}) = \mathcal{N}(\mathbf{x}_k; \sqrt{1 - \beta_k} \mathbf{x}_{k-1}, \beta_k \mathbf{I})$$

trong đó β_k ∈ (0, 1) là noise schedule (tăng dần).

**Closed-form** tại bất kỳ step k (nhờ tính chất Gaussian):

$$q(\mathbf{x}_k | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_k; \sqrt{\bar{\alpha}_k} \mathbf{x}_0, (1 - \bar{\alpha}_k) \mathbf{I})$$

trong đó:
- α_k = 1 - β_k
- ᾱ_k = Π_{s=1}^{k} α_s (cumulative product)

Hay viết dạng sampling:

$$\mathbf{x}_k = \sqrt{\bar{\alpha}_k} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Khi k → K (lớn), ᾱ_K → 0, nên **x**_K ≈ pure Gaussian noise.

**Kết nối 3DGS**: Nghĩ forward process như "blur" dần một 3D scene — ban đầu sharp, cuối cùng chỉ còn noise. Reverse process = reconstruction từ noise.

#### 3.1.2 Reverse Process (denoise)

Mục tiêu: học reverse transition p_θ(**x**_{k-1} | **x**_k):

$$p_\theta(\mathbf{x}_{k-1} | \mathbf{x}_k) = \mathcal{N}(\mathbf{x}_{k-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_k, k), \sigma_k^2 \mathbf{I})$$

Bài toán quy về: predict μ_θ(**x**_k, k).

Có 3 cách tương đương để parameterize μ_θ:

**(a) Predict noise ε_θ** (cách phổ biến nhất):

$$\boldsymbol{\mu}_\theta(\mathbf{x}_k, k) = \frac{1}{\sqrt{\alpha_k}} \left(\mathbf{x}_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_k, k)\right)$$

**(b) Predict x_0 trực tiếp**:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_k, k) = \frac{\sqrt{\bar{\alpha}_{k-1}} \beta_k}{1 - \bar{\alpha}_k} \hat{\mathbf{x}}_0 + \frac{\sqrt{\alpha_k}(1 - \bar{\alpha}_{k-1})}{1 - \bar{\alpha}_k} \mathbf{x}_k$$

**(c) Predict score ∇ log p(x_k)** (score matching):

$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_k, k) = -\sqrt{1 - \bar{\alpha}_k} \nabla_{\mathbf{x}_k} \log p(\mathbf{x}_k)$$

Cả 3 cách tương đương toán học, nhưng (a) ổn định nhất khi training.

#### 3.1.3 Training Loss

$$\mathcal{L}_{DDPM} = \mathbb{E}_{k \sim \text{Uniform}(1,K), \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I}), \mathbf{x}_0 \sim q(\mathbf{x}_0)} \left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_k, k)\|^2\right]$$

Tức là: sample k ngẫu nhiên, thêm noise vào x_0 để được x_k, rồi train network predict noise.

**Đơn giản đến bất ngờ**: Loss chỉ là MSE giữa true noise và predicted noise. Không cần ELBO phức tạp như VAE.

### 3.2 Diffusion Policy — Adapt DDPM cho Robot Actions

#### 3.2.1 Key Mapping

| DDPM concept | Diffusion Policy mapping |
|---|---|
| **x**_0 (clean data) | **A**_t = [**a**_t, **a**_{t+1}, ..., **a**_{t+T_a-1}] (action trajectory) |
| Conditioning | **O**_t = [**o**_{t-T_o+1}, ..., **o**_t] (observation history) |
| ε_θ(x_k, k) | ε_θ(**A**_t^k, k, **O**_t) (noise prediction conditioned on obs) |

Observation history T_o steps (không chỉ current step) vì manipulation cần temporal context.

#### 3.2.2 Conditional DDPM Training

$$\mathcal{L}_{DP} = \mathbb{E}_{k, \boldsymbol{\epsilon}, (\mathbf{O}_t, \mathbf{A}_t) \sim \mathcal{D}} \left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_k}\mathbf{A}_t + \sqrt{1-\bar{\alpha}_k}\boldsymbol{\epsilon},\ k,\ \mathbf{O}_t)\|^2\right]$$

So với standard DDPM, chỉ thêm **O**_t làm condition cho noise prediction network.

#### 3.2.3 Conditional DDPM Inference (Action Generation)

```
Input: observation history O_t
1. Sample A_t^K ~ N(0, I)  ∈ ℝ^{T_a × d_a}   (pure noise action trajectory)
2. For k = K, K-1, ..., 1:
     ε̂ = ε_θ(A_t^k, k, O_t)         # predict noise
     μ = (1/√α_k)(A_t^k - (β_k/√(1-ᾱ_k))·ε̂)  # compute mean
     A_t^{k-1} = μ + σ_k · z,  z ~ N(0,I) if k > 1, else z = 0
3. Output: A_t^0 = [â_t, ..., â_{t+T_a-1}]    (denoised action trajectory)
```

**Computational cost**: K forward passes qua neural network! Typical K = 100 cho DDPM, giảm xuống ~10-20 với DDIM sampler.

#### 3.2.4 Receding Horizon Control

Giống ACT temporal ensembling nhưng cách tiếp cận khác:
- Predict action trajectory dài T_a steps
- Chỉ **execute T_e < T_a** actions đầu tiên
- Re-plan từ observation mới

$$T_e = T_a / 2 \quad \text{(typical setting)}$$

Ví dụ: T_a = 16, T_e = 8 → predict 16 actions, execute 8, re-plan.

Tại sao T_e < T_a? Vì actions xa trong tương lai kém chính xác hơn. Overlap tạo smooth transitions giữa các plan liên tiếp.

### 3.3 Network Architectures

#### 3.3.1 CNN-based: 1D Temporal U-Net

Kiến trúc chính trong paper gốc. Xử lý action sequence như 1D signal:

```
Input: A_t^k ∈ ℝ^{T_a × d_a}  (noisy action trajectory, treated as 1D)
       k ∈ ℤ                   (diffusion timestep)
       O_t                     (visual observations)

Architecture:
1. Visual Encoder: ResNet-18 per camera → visual features f_v
2. FiLM conditioning: embed k via sinusoidal → γ_k, β_k
   Each residual block: h = γ_k · GroupNorm(h) + β_k
3. U-Net on action dimension:
   - Down blocks: Conv1D + ResBlock + Downsample
   - Mid block: Conv1D + ResBlock + Attention
   - Up blocks: Conv1D + ResBlock + Upsample + Skip connections
4. Cross-attention with visual features tại mid-resolution
5. Output: ε̂ ∈ ℝ^{T_a × d_a}  (predicted noise)
```

**FiLM conditioning**: Feature-wise Linear Modulation — cách inject timestep k vào network. Tương tự cách NeRF inject positional encoding, nhưng qua affine transform trên features thay vì concatenation.

#### 3.3.2 Transformer-based: Diffusion Transformer (DiT-style)

```
Input tokens:
  - Noisy action tokens: Linear(A_t^k) → [h_a^0, ..., h_a^{T_a-1}]
  - Timestep token: SinEmbed(k) → h_k  
  - Visual tokens: ResNet(images) → [h_v^1, ..., h_v^M]
  - (Optional) Joint state token: Linear(j_t) → h_j

Transformer Layers:
  For each layer l:
    1. Self-attention over all tokens
    2. AdaLN (Adaptive Layer Norm): conditioned on timestep k
       h = γ(k) · LayerNorm(h) + β(k)
    3. Feed-forward network

Output: Project action tokens back to ε̂ ∈ ℝ^{T_a × d_a}
```

### 3.4 Tại sao Diffusion xử lý Multimodality tốt?

Quay lại ví dụ gấp áo (trái vs phải):

**BC với MSE**: Minimize E[||π(o) - a||²] → π(o) = E[a|o] = trung bình → sai.

**CVAE (ACT)**: Multimodality qua z: z₁ → gấp trái, z₂ → gấp phải. Nhưng khi inference z=0, collapse về 1 mode.

**Diffusion**: Tại inference, starting point A^K là random noise. Mỗi lần sample, noise khác nhau → trajectory khác nhau. Diffusion model tự nhiên sample từ TOÀN BỘ distribution p(A|O), bao gồm cả mode gấp trái và gấp phải.

Toán học: Diffusion learns score function ∇ log p(A|O). Score function capture gradient landscape của toàn bộ distribution, bao gồm multiple modes. Khi denoise, trajectory bị "kéo" về mode gần nhất.

```
Score landscape cho bimodal distribution:
     ↗ ↗ ↗              ↖ ↖ ↖
   ↗ MODE 1 ↗    ↙ ↘    ↖ MODE 2 ↖  
     ↗ ↗ ↗    ↙     ↘    ↖ ↖ ↖
              saddle point
```

Noise ban đầu nằm bên trái saddle → converge về Mode 1. Nằm bên phải → Mode 2. Không bao giờ trung bình!

---

## PHẦN IV: SO SÁNH TOÁN HỌC SÂU

### 4.1 Expressiveness

| | BC | ACT (CVAE) | Diffusion Policy |
|---|---|---|---|
| Distribution family | Unimodal (MSE) hoặc GMM | Conditional Gaussian (qua z) | Arbitrary (implicit) |
| Mode coverage | 1 mode | Lý thuyết: nhiều modes. Thực tế: 1 mode khi z=0 | Tất cả modes |
| Representation | π(o) = μ | p(a\|o) = ∫ p(a\|z,o)p(z)dz | p(a\|o) via score ∇log p |

### 4.2 Training Objective Comparison

**BC**: 
$$\min_\theta \mathbb{E}[\|\mathbf{a} - f_\theta(\mathbf{o})\|^2]$$

**ACT**:
$$\min_{\theta,\phi} \mathbb{E}\left[\sum_i \|\mathbf{a}_i - \hat{\mathbf{a}}_i\|_1 + \beta \cdot D_{KL}(q_\phi(\mathbf{z}|\mathbf{a},\mathbf{o}) \| \mathcal{N}(\mathbf{0},\mathbf{I}))\right]$$

**Diffusion Policy**:
$$\min_\theta \mathbb{E}_{k,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_k}\mathbf{A} + \sqrt{1-\bar{\alpha}_k}\boldsymbol{\epsilon}, k, \mathbf{O})\|^2\right]$$

**Observations:**
- BC loss đo lỗi trực tiếp trong action space
- ACT loss đo lỗi trong action space + regularization trong latent space  
- Diffusion loss đo lỗi trong **noise space** — không trực tiếp optimize action quality!

Điều này giải thích tại sao Diffusion training ổn định: loss luôn bounded vì ε ~ N(0,I) → ||ε||² ~ χ²(d). Không bị exploding gradients do large action errors.

### 4.3 Inference Computational Cost

**BC**: 1 forward pass → O(1)

**ACT**: 1 forward pass (z=0, no encoder) → O(1)

**Diffusion Policy**: K forward passes (denoising steps) → O(K)
- K = 100 cho DDPM
- K = 10-20 cho DDIM (accelerated sampler)
- K = 1-4 cho Consistency Models / Flow Matching (state-of-the-art)

Cho LeHome với SO-ARM101 control frequency ~50Hz → cần inference trong ~20ms. ACT: dễ dàng. Diffusion: cần DDIM hoặc fast sampler.

### 4.4 Action Chunking: Shared Insight

Cả ACT và Diffusion Policy đều predict **chuỗi actions** thay vì single action.

Effective horizon reduction:

$$T_{eff} = \lceil T / k \rceil$$

trong đó k = chunk size (ACT) hoặc T_a (Diffusion Policy).

Compounding error bound giảm từ O(εT²) → O(ε T_{eff}²) = O(ε(T/k)²).

Với T=200, k=20: từ O(40000ε) → O(100ε). Giảm 400x!

---

## PHẦN V: IMPLEMENTATION NOTES CHO LEROBOT

### 5.1 ACT trong LeRobot

```python
# Key hyperparameters
chunk_size = 100        # k: số actions per chunk
kl_weight = 10          # β: KL regularization weight  
hidden_dim = 512        # transformer hidden dimension
dim_feedforward = 3200  # FFN dimension
n_heads = 8             # attention heads
n_encoder_layers = 4    # CVAE encoder layers
n_decoder_layers = 7    # policy decoder layers
latent_dim = 32         # d_z: latent variable dimension
```

**Tuning tips cho garment manipulation:**
- Tăng chunk_size nếu garment tasks dài (folding = many steps)
- Giảm kl_weight nếu thấy policy quá "boring" (posterior collapse → z uninformative)
- Thêm cameras → nhiều visual tokens → tốn memory, cân nhắc giảm image resolution

### 5.2 Diffusion Policy trong LeRobot

```python
# Key hyperparameters
n_action_steps = 8       # T_e: actions to execute
horizon = 16             # T_a: prediction horizon
n_obs_steps = 2          # T_o: observation history
num_inference_steps = 10 # K: denoising steps (DDIM)
noise_scheduler = "ddim" # DDIM for faster inference
```

**Tuning tips:**
- num_inference_steps: 10 thường đủ cho DDIM. Giảm nếu cần faster control.
- horizon vs n_action_steps ratio: 2:1 là sweet spot. Tăng horizon nếu task cần longer planning.
- noise_scheduler: DDIM >> DDPM cho inference speed. Flow matching nếu LeRobot hỗ trợ.

---

## PHẦN VI: GỢI Ý NGHIÊN CỨU CHO LEHOME

### 6.1 Hypothesis: 3D Representation + Diffusion Policy cho Cloth

Background 3DGS của bạn có thể exploit bằng cách:
1. Dùng dual cameras (gripper + external) để estimate cloth 3D structure
2. Feed 3D features (point cloud hoặc depth) vào Diffusion Policy thay vì chỉ 2D images
3. Tham khảo DP3 (3D Diffusion Policy) paper

### 6.2 Curriculum: Bắt đầu thế nào

1. **Ngày 1-3**: Chạy ACT baseline trên LeHome simulation dataset
2. **Ngày 4-6**: Chạy Diffusion Policy baseline. So sánh với ACT.
3. **Ngày 7-14**: Improve visual backbone (DINOv2), tune hyperparameters
4. **Ngày 15+**: Thử 3D conditioning, data augmentation, custom losses
