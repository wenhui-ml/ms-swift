from swift.model.mag_gated.configuration_mag_gated import MagGatedConfig
from swift.model.mag_gated.modeling_mag_gated import MagGatedForCausalLM, ResidualGate
import torch

config = MagGatedConfig(
    hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    vocab_size=100, use_residual_gate=True,
    residual_gate_n_groups=4,  # 64/4=16 dims per group
    residual_gate_init_bias=5.0,
)
model = MagGatedForCausalLM(config)

# Verify all projections are pure nn.Linear
layer0 = model.model.layers[0]
for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    assert type(getattr(layer0.self_attn, name)).__name__ == 'Linear'
for name in ['gate_proj', 'up_proj', 'down_proj']:
    assert type(getattr(layer0.mlp, name)).__name__ == 'Linear'
print('OK: All projections are pure nn.Linear')

# Verify V5 grouped attention gate structure
gate = layer0.attn_residual_gate
assert isinstance(gate, ResidualGate)
assert gate.n_groups == 4
assert gate.group_size == 16
assert gate.w_q.shape == (4, 16), f'Expected (4,16), got {gate.w_q.shape}'
assert gate.w_kh.shape == (64,), f'Expected (64,), got {gate.w_kh.shape}'
assert gate.w_ko.shape == (64,), f'Expected (64,), got {gate.w_ko.shape}'
assert gate.b_forget.shape == (4,), f'Expected (4,), got {gate.b_forget.shape}'
assert gate.b_accept.shape == (4,), f'Expected (4,), got {gate.b_accept.shape}'
print('OK: V5 grouped attention gate has correct shapes')

# Verify initialization
assert gate.w_q.data.abs().max().item() < 1e-6, 'w_q should be zero'
assert (gate.w_kh.data - 1.0).abs().max().item() < 1e-6, 'w_kh should be 1'
forget_init = torch.sigmoid(gate.b_forget.data.float().mean()).item()
accept_init = torch.sigmoid(gate.b_accept.data.float().mean()).item()
print(f'OK: init forget={forget_init:.4f} (≈0), accept={accept_init:.4f} (≈1)')
print(f'    → h_new ≈ (1-{forget_init:.4f})h + {accept_init:.4f}o ≈ h + o')
assert forget_init < 0.01, f'forget should be near 0, got {forget_init}'
assert accept_init > 0.99, f'accept should be near 1, got {accept_init}'

# Forward pass
x = torch.randint(0, 100, (1, 8))
with torch.no_grad():
    out = model(x)
print(f'OK: logits shape={out.logits.shape}')

# Verify gradient flows through gate
model.train()
model.zero_grad()
x = torch.randint(0, 100, (1, 8))
out = model(x, labels=x)
out.loss.backward()
assert gate.w_q.grad is not None, 'w_q should have gradient'
assert gate.w_kh.grad is not None, 'w_kh should have gradient'
assert gate.b_forget.grad is not None, 'b_forget should have gradient'
assert gate.b_accept.grad is not None, 'b_accept should have gradient'
print(f'OK: All gate parameters receive gradients')
print(f'    w_q grad norm={gate.w_q.grad.norm():.6f}')
print(f'    w_kh grad norm={gate.w_kh.grad.norm():.6f}')

# Verify near-identity: h_new ≈ h + o at initialization
model.eval()
h = torch.randn(1, 4, 64)
o = torch.randn(1, 4, 64)
with torch.no_grad():
    h_new = gate(h, o)
    h_standard = h + o
    diff = (h_new - h_standard).abs().max().item()
print(f'OK: |h_new - (h+o)| max = {diff:.6f} (should be near 0)')
assert diff < 0.05, f'Near-identity check failed: diff={diff}'

# Count gate parameters
gate_params = sum(p.numel() for n, p in model.named_parameters() if 'residual_gate' in n)
total_params = sum(p.numel() for p in model.parameters())
print(f'OK: Gate params={gate_params:,} ({gate_params/total_params*100:.2f}% of total {total_params:,})')

print()
print('=' * 60)
print('ALL TESTS PASSED ✓')
print('=' * 60)
