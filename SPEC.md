# mHC Implementation Specification for Qwen3-0.6B

## Document Version: 1.0
## Target: Complete architectural specification for implementation

---

## 1. Executive Summary

This specification describes how to expand Qwen3-0.6B using Manifold-Constrained Hyper-Connections (mHC). The implementation transforms the single-stream residual architecture into a 4-stream architecture with learnable inter-stream connections, while maintaining mathematical equivalence at initialization.

**Key Deliverables:**
1. mHC layer module that wraps Transformer sublayers
2. Sinkhorn-Knopp algorithm for doubly stochastic projection
3. Weight conversion utility from standard Qwen3 to mHC
4. Equivalence validation tooling

---

## 2. Base Model: Qwen3-0.6B Architecture

### 2.1 Confirmed Configuration (from HuggingFace)

```
hidden_size (C):           1024
intermediate_size:         3072
num_hidden_layers:         28
num_attention_heads:       16
num_key_value_heads:       8
head_dim:                  128
vocab_size:                151936
max_position_embeddings:   40960
hidden_act:                silu
rms_norm_eps:              1e-6
tie_word_embeddings:       true
rope_theta:                1000000
```

### 2.2 Layer Structure

Each of the 28 decoder layers contains:
- `input_layernorm` → `self_attn` → residual add
- `post_attention_layernorm` → `mlp` → residual add

The mHC paper treats Attention and MLP as separate "layers" for mHC purposes, giving us **56 total sublayers** (28 × 2).

---

## 3. mHC Architecture Specification

### 3.1 Core Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Expansion rate | n | 4 | Number of parallel residual streams |
| Hidden size | C | 1024 | Original model dimension |
| Expanded size | n×C | 4096 | New residual stream width |
| Gating init | α | 0.01 | Initial value for learnable gates |
| Sinkhorn iterations | t_max | 20 | Iterations for doubly stochastic projection |

### 3.2 Tensor Shape Transformations

**Standard Qwen3:**
```
Input:  (Batch, Sequence, 1024)
Output: (Batch, Sequence, 1024)
```

**mHC Qwen3:**
```
Input to model:     (Batch, Sequence, 1024)
After expansion:    (Batch, Sequence, 4, 1024)
Throughout layers:  (Batch, Sequence, 4, 1024)
Before LM head:     (Batch, Sequence, 4, 1024)
After collapse:     (Batch, Sequence, 1024)
```

### 3.3 Three Learnable Mappings

For each sublayer l, mHC introduces three mapping matrices:

#### H_pre (Pre-Mapping / Squeeze)
- **Purpose**: Aggregate n streams into 1 for the sublayer input
- **Shape**: (Batch, Sequence, 1, n) or static (1, n)
- **Operation**: Matrix multiply to squeeze n×C → C
- **Constraint**: Sigmoid applied (values in [0, 1])

#### H_post (Post-Mapping / Broadcast)  
- **Purpose**: Distribute sublayer output back to n streams
- **Shape**: (Batch, Sequence, 1, n) or static (1, n)
- **Operation**: Outer product to expand C → n×C
- **Constraint**: 2×Sigmoid applied (values in [0, 2])

#### H_res (Residual-Mapping / Mixer)
- **Purpose**: Mix information between the n streams
- **Shape**: (Batch, Sequence, n, n) or static (n, n)
- **Operation**: Matrix multiply within the stream dimension
- **Constraint**: **Doubly stochastic** via Sinkhorn-Knopp (rows and columns sum to 1)

### 3.4 Forward Pass Algorithm

For each mHC-wrapped sublayer:

```
INPUT: x ∈ ℝ^(B, S, n, C)  -- expanded hidden state

1. FLATTEN for coefficient computation:
   x_flat = reshape(x) → (B, S, n×C)

2. COMPUTE COEFFICIENTS (see Section 4):
   H_pre, H_post, H_res = compute_coefficients(x_flat)

3. PRE-MAPPING (squeeze n streams to 1):
   h_in = H_pre @ x           → (B, S, 1, C)
   h_in = squeeze(h_in)       → (B, S, C)

4. APPLY SUBLAYER (original Attention or MLP):
   h_out = LayerNorm(h_in)
   h_out = Sublayer(h_out)    → (B, S, C)

5. POST-MAPPING (broadcast 1 stream to n):
   h_out = unsqueeze(h_out)   → (B, S, 1, C)
   h_post = H_post.T @ h_out  → (B, S, n, C)

6. RESIDUAL-MAPPING (mix streams):
   h_res = H_res @ x          → (B, S, n, C)

7. COMBINE:
   output = h_res + h_post    → (B, S, n, C)

OUTPUT: output ∈ ℝ^(B, S, n, C)
```

---

## 4. Coefficient Computation

### 4.1 Dynamic vs Static Components

Each mapping has both dynamic (input-dependent) and static (learned bias) components:

```
H̃ = α · dynamic(x) + bias_static
```

Where:
- `α` is a learnable scalar gate (initialized to 0.01)
- `dynamic(x)` is computed from the current hidden state
- `bias_static` is a learned bias term

### 4.2 Dynamic Mapping Computation

```
1. NORMALIZE the flattened hidden state:
   x' = RMSNorm(x_flat)     -- shape (B, S, n×C)

2. PROJECT to coefficient space:
   H̃_pre_dynamic  = x' @ φ_pre    -- φ_pre  ∈ ℝ^(n×C, n)   → result (B, S, n)
   H̃_post_dynamic = x' @ φ_post   -- φ_post ∈ ℝ^(n×C, n)   → result (B, S, n)
   H̃_res_dynamic  = x' @ φ_res    -- φ_res  ∈ ℝ^(n×C, n²)  → result (B, S, n²)
   
3. RESHAPE residual mapping:
   H̃_res_dynamic = reshape(H̃_res_dynamic) → (B, S, n, n)

4. COMBINE with static bias:
   H̃_pre  = α_pre  · H̃_pre_dynamic  + b_pre
   H̃_post = α_post · H̃_post_dynamic + b_post
   H̃_res  = α_res  · H̃_res_dynamic  + b_res
```

### 4.3 Constraint Application

```
H_pre  = sigmoid(H̃_pre)              -- Non-negative, bounded [0,1]
H_post = 2 × sigmoid(H̃_post)         -- Non-negative, bounded [0,2]
H_res  = SinkhornKnopp(H̃_res)        -- Doubly stochastic
```

### 4.4 Parameter Count per Sublayer

| Parameter | Shape | Count |
|-----------|-------|-------|
| φ_pre | (4096, 4) | 16,384 |
| φ_post | (4096, 4) | 16,384 |
| φ_res | (4096, 16) | 65,536 |
| b_pre | (1, 4) | 4 |
| b_post | (1, 4) | 4 |
| b_res | (4, 4) | 16 |
| α_pre, α_post, α_res | scalars | 3 |
| RMSNorm weight | (4096,) | 4,096 |
| **Total per sublayer** | | **102,427** |
| **Total for model (56 sublayers)** | | **~5.7M** |

---

## 5. Sinkhorn-Knopp Algorithm

### 5.1 Purpose

Projects an arbitrary matrix onto the **Birkhoff polytope** (set of doubly stochastic matrices where all rows and columns sum to 1).

### 5.2 Properties of Doubly Stochastic Matrices

1. **Norm bounded**: Spectral norm ≤ 1 (non-expansive)
2. **Closure under multiplication**: Product of DS matrices is DS
3. **Convex combination**: Acts as weighted average preserving total signal

These properties ensure that the product of H_res matrices across many layers remains bounded, preventing signal explosion/vanishing.

### 5.3 Algorithm

```
INPUT: M ∈ ℝ^(n, n) -- unconstrained matrix

1. EXPONENTIATE to make positive:
   H = exp(M)              -- All entries now > 0

2. ITERATE alternating normalizations (t_max = 20 times):
   FOR t = 1 to t_max:
       H = H / row_sum(H)   -- Normalize rows to sum to 1
       H = H / col_sum(H)   -- Normalize columns to sum to 1

OUTPUT: H ∈ ℝ^(n, n) -- doubly stochastic matrix
```

### 5.4 Numerical Considerations

- Add small epsilon (1e-8) to denominators to prevent division by zero
- Compute in float32 for numerical stability, even if model uses bfloat16
- 20 iterations achieves error < 1e-10 for n=4

### 5.5 Backward Pass

The gradient through Sinkhorn-Knopp can be computed by:
1. Recomputing forward pass during backward (saves memory)
2. Using implicit differentiation through the fixed-point equation
3. The gradient accounts for the doubly stochastic constraint

---

## 6. Initialization for Equivalence

### 6.1 Goal

Initialize all mHC parameters such that the expanded model produces **identical outputs** to the original Qwen3-0.6B model.

### 6.2 Equivalence Conditions

For the mHC model to be equivalent to standard residual:
- H_pre must **average** the n streams (since they're copies, average = original)
- H_post must **copy** the output to all n streams equally
- H_res must be the **identity matrix** (no mixing)

### 6.3 H_pre Initialization

**Target**: After sigmoid, each entry should be 1/n = 0.25

```
sigmoid(x) = 1/n = 0.25
x = logit(0.25) = log(0.25 / 0.75) = log(1/3) ≈ -1.0986

Initialize: b_pre = [-1.0986, -1.0986, -1.0986, -1.0986]
```

**Verification**: sigmoid(-1.0986) ≈ 0.25, so H_pre @ x averages the 4 streams.

### 6.4 H_post Initialization

**Target**: After 2×sigmoid, each entry should be 1.0

```
2 × sigmoid(x) = 1.0
sigmoid(x) = 0.5
x = logit(0.5) = 0

Initialize: b_post = [0, 0, 0, 0]
```

**Verification**: 2 × sigmoid(0) = 1.0, so output is copied equally to all streams.

### 6.5 H_res Initialization

**Target**: After Sinkhorn-Knopp, result should be identity matrix I_4

```
Sinkhorn-Knopp(exp(M)) → I when M has large diagonal

Initialize: b_res = [[20, 0, 0, 0],
                     [0, 20, 0, 0],
                     [0, 0, 20, 0],
                     [0, 0, 0, 20]]
```

**Verification**: exp(20) >> exp(0), so after normalization, diagonal dominates → identity.

### 6.6 Dynamic Mapping Initialization

- All φ_pre, φ_post, φ_res weights: **Initialize to zero**
- All α gates: **Initialize to 0.01**

This ensures dynamic components contribute nothing initially (α × 0 = 0).

### 6.7 Stream Expansion/Collapse

**At model input** (after embedding):
- Expand (B, S, C) → (B, S, n, C) by **copying** the hidden state n times

**At model output** (before LM head):
- Collapse (B, S, n, C) → (B, S, C) by **averaging** the n streams

---

## 7. Model Architecture Modifications

### 7.1 New Components to Add

1. **StreamExpand module**: Placed after embedding layer
2. **MHCLayer wrapper**: Wraps each Attention and MLP sublayer (56 total)
3. **StreamCollapse module**: Placed before final LayerNorm and LM head

### 7.2 Modified Forward Flow

```
Original Qwen3:
  Embedding → [DecoderLayer × 28] → FinalNorm → LMHead

mHC Qwen3:
  Embedding → StreamExpand → [MHCDecoderLayer × 28] → StreamCollapse → FinalNorm → LMHead
```

### 7.3 MHCDecoderLayer Structure

Each MHCDecoderLayer contains:
```
MHCDecoderLayer:
├── mhc_attn: MHCLayer
│   ├── coef_norm: RMSNorm(4096)
│   ├── phi_pre, phi_post, phi_res: Linear projections
│   ├── b_pre, b_post, b_res: Bias parameters
│   ├── alpha_pre, alpha_post, alpha_res: Gate scalars
│   └── sublayer: [input_layernorm + self_attn]
│
└── mhc_mlp: MHCLayer
    ├── coef_norm: RMSNorm(4096)
    ├── phi_pre, phi_post, phi_res: Linear projections
    ├── b_pre, b_post, b_res: Bias parameters
    ├── alpha_pre, alpha_post, alpha_res: Gate scalars
    └── sublayer: [post_attention_layernorm + mlp]
```

### 7.4 Weight Mapping

| Original Component | mHC Location | Notes |
|--------------------|--------------|-------|
| model.embed_tokens | model.embed_tokens | Unchanged |
| model.layers[i].input_layernorm | model.layers[i].mhc_attn.sublayer.layernorm | Unchanged weights |
| model.layers[i].self_attn | model.layers[i].mhc_attn.sublayer.attention | Unchanged weights |
| model.layers[i].post_attention_layernorm | model.layers[i].mhc_mlp.sublayer.layernorm | Unchanged weights |
| model.layers[i].mlp | model.layers[i].mhc_mlp.sublayer.mlp | Unchanged weights |
| model.norm | model.norm | Unchanged |
| lm_head | lm_head | Unchanged (tied) |

---

## 8. Training Specifications

### 8.1 Recommended Hyperparameters

Based on the mHC paper's 3B model settings (scaled for 0.6B):

| Parameter | Value |
|-----------|-------|
| Learning rate | 8.6e-4 |
| Batch size | 320 |
| Sequence length | 4096 |
| Training steps | 30,000 |
| Warmup steps | 2,000 |
| Optimizer | AdamW |
| Adam β1, β2 | 0.9, 0.95 |
| Adam ε | 1e-20 |
| Weight decay | 0.1 |
| LR schedule | Step decay at 80% and 90% of training |
| Decay multipliers | 0.316, then 0.1 |

### 8.2 Gradient Checkpointing Strategy

The n-stream design increases activation memory ~4×. Mitigation:

1. **Checkpoint every L_r layers** where L_r ≈ √(n×L/(n+2))
   - For n=4, L=28: L_r ≈ √(4×28/6) ≈ 4
   
2. **Recompute mHC coefficients** during backward pass
   - Store only: x_l at checkpoint boundaries, sublayer outputs
   - Recompute: H_pre, H_post, H_res from stored x_l

### 8.3 Stability Monitoring

Track these metrics during training:

1. **Gradient norm**: Should stay stable, spikes indicate instability
2. **Forward gain**: max(row_sum(H_res)) should stay near 1.0
3. **Backward gain**: max(col_sum(H_res)) should stay near 1.0
4. **Identity distance**: ||H_res - I||_F shows how much mixing is learned

**Warning thresholds**:
- Gradient norm > 10: Potential explosion
- Forward/backward gain > 2: Instability risk
- Loss increase > 50%: Training collapse

---

## 9. Memory and Compute Estimates

### 9.1 Parameter Overhead

| Component | Original | mHC Addition |
|-----------|----------|--------------|
| Embedding | 155.6M | 0 |
| Attention layers | 176.2M | 0 |
| MLP layers | 215.0M | 0 |
| mHC parameters | 0 | 5.7M |
| **Total** | ~600M | +5.7M (+1%) |

### 9.2 Activation Memory Overhead

| Component | Original | mHC |
|-----------|----------|-----|
| Hidden states per layer | C = 1024 | n×C = 4096 |
| Memory multiplier | 1× | ~4× |
| With checkpointing | 1× | ~1.5× |

### 9.3 Compute Overhead

| Operation | FLOPs (per token, per layer) |
|-----------|------------------------------|
| Original layer | ~12.6M |
| mHC coefficient computation | ~0.4M |
| mHC matrix operations | ~0.03M |
| **Overhead** | ~3% |

Paper reports ~6.7% wall-clock overhead with optimized kernels.

---

## 10. Testing Requirements

### 10.1 Unit Tests

1. **Sinkhorn-Knopp correctness**
   - Output is doubly stochastic (rows/cols sum to 1)
   - Convergence within 20 iterations
   - Gradient flows correctly

2. **MHCLayer correctness**
   - Output shapes match expected
   - Equivalence initialization produces identity behavior
   - Gradients flow to all parameters

3. **Coefficient computation**
   - RMSNorm applied correctly
   - Dynamic and static components combine properly
   - Constraints applied correctly

### 10.2 Integration Tests

1. **Weight conversion**
   - All original weights transferred correctly
   - mHC parameters initialized for equivalence
   - Model loads and runs without error

2. **Equivalence validation**
   - Original and mHC models produce identical logits (within tolerance)
   - Test with multiple input sequences
   - Test with different batch sizes

3. **Training stability**
   - Loss decreases over 1000 steps
   - Gradient norms remain bounded
   - No NaN/Inf values

### 10.3 Acceptance Criteria

| Test | Criterion |
|------|-----------|
| Equivalence | Max logit difference < 1e-5 |
| Sinkhorn convergence | Row/col sum error < 1e-8 |
| Training stability | No loss spikes > 2× in 10k steps |
| Memory overhead | < 2× with checkpointing |

---

## 11. File Structure

```
mhc/
├── README.md                      # Overview and quick start
├── SPEC.md                        # This specification
├── requirements.txt               # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # MHCConfig dataclass
│   ├── sinkhorn.py                # Sinkhorn-Knopp implementation
│   ├── mhc_layer.py               # MHCLayer module
│   ├── stream_ops.py              # StreamExpand, StreamCollapse
│   ├── qwen3_mhc_model.py         # Full model with mHC
│   └── conversion.py              # Weight conversion utilities
│
├── scripts/
│   ├── convert_to_mhc.py          # CLI for conversion
│   ├── validate_equivalence.py    # Equivalence testing
│   └── train.py                   # Training script
│
├── tests/
│   ├── test_sinkhorn.py
│   ├── test_mhc_layer.py
│   ├── test_conversion.py
│   └── test_equivalence.py
│
└── configs/
    └── qwen3_0.6b_mhc.yaml        # Training config
```

---

## 12. Dependencies

```
torch >= 2.0
transformers >= 4.51.0
safetensors
accelerate
pyyaml
pytest (dev)
```

---

## 13. References

1. **mHC Paper**: "mHC: Manifold-Constrained Hyper-Connections" - DeepSeek-AI, arXiv:2512.24880, Dec 2025
2. **Original HC Paper**: "Hyper-Connections" - Zhu et al., arXiv:2409.19606, 2024
3. **Qwen3 Technical Report**: arXiv:2505.09388, May 2025
4. **Sinkhorn-Knopp**: Sinkhorn & Knopp, "Concerning nonnegative matrices and doubly stochastic matrices", Pacific Journal of Mathematics, 1967

---

## Appendix A: Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| S | Sequence length |
| C | Hidden dimension (1024) |
| n | Expansion rate (4) |
| L | Number of sublayers (56) |
| x_l | Hidden state at layer l, shape (B, S, n, C) |
| H_pre | Pre-mapping coefficients |
| H_post | Post-mapping coefficients |
| H_res | Residual mixing matrix (doubly stochastic) |
| φ | Dynamic projection weights |
| b | Static bias terms |
| α | Learnable gating scalars |
| P_M | Manifold projection (Sinkhorn-Knopp) |

---

## Appendix B: Equivalence Proof Sketch

At initialization with the specified values:

1. **H_pre** = [0.25, 0.25, 0.25, 0.25] (after sigmoid)
2. **H_post** = [1, 1, 1, 1] (after 2×sigmoid)  
3. **H_res** = I_4 (after Sinkhorn on large diagonal)

For input x with 4 identical streams [v, v, v, v]:

```
h_in = H_pre @ x = 0.25×v + 0.25×v + 0.25×v + 0.25×v = v  ✓
h_out = Sublayer(v) = v'
h_post = H_post.T @ v' = [v', v', v', v']  ✓
h_res = H_res @ x = I @ [v,v,v,v] = [v, v, v, v]  ✓
output = h_res + h_post = [v+v', v+v', v+v', v+v']  ✓
```

After collapsing by average: (v+v') = original residual output. QED.

---

## Appendix C: Key Differences from Original HC

| Aspect | HC | mHC |
|--------|-----|------|
| H_res constraint | None (unconstrained) | Doubly stochastic |
| H_pre/H_post | tanh activation | Sigmoid activation |
| Stability | Prone to explosion | Bounded by design |
| Composite mapping | Unbounded product | Product remains DS |
| Maximum gain | Can be arbitrarily large | Bounded by 1 |
