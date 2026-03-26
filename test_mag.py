from swift.model.mag_gated.configuration_mag_gated import MagGatedConfig
from swift.model.mag_gated.modeling_mag_gated import MagGatedForCausalLM
import torch

config = MagGatedConfig(
    hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    vocab_size=100, use_residual_gate=True, residual_gate_rank=4,
    residual_gate_init_bias=5.0,
)
model = MagGatedForCausalLM(config)

layer0 = model.model.layers[0]
for name in ['q_proj','k_proj','v_proj','o_proj']:
    assert type(getattr(layer0.self_attn, name)).__name__ == 'Linear'
for name in ['gate_proj','up_proj','down_proj']:
    assert type(getattr(layer0.mlp, name)).__name__ == 'Linear'
print('OK: All projections are pure nn.Linear')

gate = layer0.attn_residual_gate
ab = gate.gate_B_alpha.bias.data.float().mean().item()
bb = gate.gate_B_beta.bias.data.float().mean().item()
print(f'OK: init_bias alpha={ab:.1f} beta={bb:.1f} (sigmoid={torch.sigmoid(torch.tensor(5.0)).item():.4f})')

x = torch.randint(0, 100, (1, 8))
with torch.no_grad():
    out = model(x)
print(f'OK: logits shape={out.logits.shape}')

print('ALL TESTS PASSED')
