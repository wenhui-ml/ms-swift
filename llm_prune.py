import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

original_model_path = "/home/ubuntu/llm_weights/Qwen3-0.6B"
output_path = "./Qwen3-LLMPruner-0.5"

# ==========================================
# 🎯 设定保留比例：0.5 (减半) 或 0.25 (减四分之三)
# ==========================================
KEEP_RATIO = 0.5  

# 1. 加载模型
print("正在加载原始模型...")
config = AutoConfig.from_pretrained(original_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(original_model_path, trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(original_model_path, trust_remote_code=True)

# 2. 计算新维度
old_hidden = config.hidden_size
old_inter = config.intermediate_size
old_heads = config.num_attention_heads
old_kv_heads = getattr(config, "num_key_value_heads", old_heads)

new_hidden = int(old_hidden * KEEP_RATIO)
new_inter = int(old_inter * KEEP_RATIO)
new_heads = int(old_heads * KEEP_RATIO)
new_kv_heads = max(1, int(old_kv_heads * KEEP_RATIO))

# ==========================================
# 🛠️ 核心修改 1：暴力扫荡，覆盖所有隐藏别名配置
# ==========================================
for key in list(config.__dict__.keys()):
    val = getattr(config, key)
    if val == old_hidden:
        setattr(config, key, new_hidden)
    elif val == old_inter:
        setattr(config, key, new_inter)
    elif val == old_heads:
        setattr(config, key, new_heads)
    elif val == old_kv_heads:
        setattr(config, key, new_kv_heads)

# ==========================================
# 🧠 评估全局 Hidden Size 通道重要性
# ==========================================
print("正在评估 Hidden Size 通道重要性...")
hidden_importance = model.model.embed_tokens.weight.norm(p=2, dim=0) + \
                    model.lm_head.weight.norm(p=2, dim=0)

for layer in model.model.layers:
    hidden_importance += layer.input_layernorm.weight.abs()
    hidden_importance += layer.post_attention_layernorm.weight.abs()

_, global_hidden_idx = torch.topk(hidden_importance, new_hidden)
global_hidden_idx = global_hidden_idx.sort()[0]

# 创建新模型
new_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# ==========================================
# ✂️ 执行重要性索引剪枝 (动态自适应版)
# ==========================================
print("正在基于 L2 Magnitude 执行结构化剪枝...")
with torch.no_grad():
    new_model.model.embed_tokens.weight.copy_(model.model.embed_tokens.weight.index_select(1, global_hidden_idx))
    new_model.lm_head.weight.copy_(model.lm_head.weight.index_select(1, global_hidden_idx))
    new_model.model.norm.weight.copy_(model.model.norm.weight.index_select(0, global_hidden_idx))
    
    for i in range(len(model.model.layers)):
        orig = model.model.layers[i]
        new = new_model.model.layers[i]
        
        # --- A. LayerNorm ---
        new.input_layernorm.weight.copy_(orig.input_layernorm.weight.index_select(0, global_hidden_idx))
        new.post_attention_layernorm.weight.copy_(orig.post_attention_layernorm.weight.index_select(0, global_hidden_idx))
        
        # --- B. MLP ---
        mlp_imp = orig.mlp.gate_proj.weight.norm(p=2, dim=1) + \
                  orig.mlp.up_proj.weight.norm(p=2, dim=1) + \
                  orig.mlp.down_proj.weight.norm(p=2, dim=0)
        _, inter_idx = torch.topk(mlp_imp, new_inter)
        inter_idx = inter_idx.sort()[0]
        
        new.mlp.gate_proj.weight.copy_(orig.mlp.gate_proj.weight.index_select(0, inter_idx).index_select(1, global_hidden_idx))
        new.mlp.up_proj.weight.copy_(orig.mlp.up_proj.weight.index_select(0, inter_idx).index_select(1, global_hidden_idx))
        new.mlp.down_proj.weight.copy_(orig.mlp.down_proj.weight.index_select(1, inter_idx).index_select(0, global_hidden_idx))

        # --- C. Attention (核心修改 2：动态追踪真实大小) ---
        new_q_out = new.self_attn.q_proj.weight.shape[0]
        new_kv_out = new.self_attn.k_proj.weight.shape[0]
        new_o_in = new.self_attn.o_proj.weight.shape[1]
        
        new.self_attn.q_proj.weight.copy_(orig.self_attn.q_proj.weight[:new_q_out, :].index_select(1, global_hidden_idx))
        new.self_attn.k_proj.weight.copy_(orig.self_attn.k_proj.weight[:new_kv_out, :].index_select(1, global_hidden_idx))
        new.self_attn.v_proj.weight.copy_(orig.self_attn.v_proj.weight[:new_kv_out, :].index_select(1, global_hidden_idx))
        new.self_attn.o_proj.weight.copy_(orig.self_attn.o_proj.weight.index_select(0, global_hidden_idx)[:, :new_o_in])

        # 【防报错补丁】如果 Qwen 模型带有 Bias 参数，也同步切片
        if getattr(orig.self_attn.q_proj, "bias", None) is not None:
            new.self_attn.q_proj.bias.copy_(orig.self_attn.q_proj.bias[:new_q_out])
            new.self_attn.k_proj.bias.copy_(orig.self_attn.k_proj.bias[:new_kv_out])
            new.self_attn.v_proj.bias.copy_(orig.self_attn.v_proj.bias[:new_kv_out])
            new.self_attn.o_proj.bias.copy_(orig.self_attn.o_proj.bias.index_select(0, global_hidden_idx))

# 5. 保存
new_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"✅ Magnitude-based LLM-Pruner 模型已成功保存至: {output_path}")