#!/bin/bash
# ============================================================================
# 将 MagGated checkpoint 准备为可独立加载的格式
#
# 问题：MagGated 是自定义架构，checkpoint 目录中不包含 Python 模型代码。
#   lm_eval / swift eval 等工具用 AutoConfig.from_pretrained() 加载时，
#   即使设了 --trust_remote_code，也找不到 model_type="mag_gated" 对应的类。
#
# 解决方案：将 modeling/configuration 代码复制到 checkpoint 目录中，
#   并在 config.json 中添加 auto_map 字段，使 HuggingFace AutoClasses
#   能自动发现并加载自定义模型。
#
# Usage:
#   bash configs/prepare_checkpoint_for_eval.sh <checkpoint_path>
#   bash configs/prepare_checkpoint_for_eval.sh output_bm/mag_gated-d1024-L28/v0-20260327-071047/checkpoint-300
#
# 批量处理所有 checkpoint：
#   for ckpt in output_bm/*/v0-*/checkpoint-*; do
#       bash configs/prepare_checkpoint_for_eval.sh "$ckpt"
#   done
#
# 处理后可直接使用：
#   lm_eval --model hf --trust_remote_code --model_args pretrained=<checkpoint_path> ...
# ============================================================================

set -e

CKPT_PATH=${1:?Usage: bash configs/prepare_checkpoint_for_eval.sh <checkpoint_path>}

# 找到模型代码源目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWIFT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_CODE_DIR="$SWIFT_ROOT/swift/model/mag_gated"

if [ ! -f "$MODEL_CODE_DIR/modeling_mag_gated.py" ]; then
    echo "ERROR: Cannot find model code at $MODEL_CODE_DIR"
    exit 1
fi

if [ ! -f "$CKPT_PATH/config.json" ]; then
    echo "ERROR: No config.json found at $CKPT_PATH"
    exit 1
fi

echo "============================================"
echo "Preparing checkpoint for standalone loading"
echo "  Checkpoint: $CKPT_PATH"
echo "  Model code: $MODEL_CODE_DIR"
echo "============================================"

# 1. 复制模型代码到 checkpoint 目录
cp "$MODEL_CODE_DIR/configuration_mag_gated.py" "$CKPT_PATH/"
cp "$MODEL_CODE_DIR/modeling_mag_gated.py" "$CKPT_PATH/"

# 2. 修复 modeling_mag_gated.py 中的相对导入为本地导入
#    原始文件: from .configuration_mag_gated import MagGatedConfig
#    需要改为: from configuration_mag_gated import MagGatedConfig
sed -i 's/from \.configuration_mag_gated import/from configuration_mag_gated import/' \
    "$CKPT_PATH/modeling_mag_gated.py"

# 3. 在 config.json 中添加 auto_map（如果不存在）
if ! grep -q "auto_map" "$CKPT_PATH/config.json"; then
    python3 -c "
import json
with open('$CKPT_PATH/config.json', 'r') as f:
    config = json.load(f)
config['auto_map'] = {
    'AutoConfig': 'configuration_mag_gated.MagGatedConfig',
    'AutoModelForCausalLM': 'modeling_mag_gated.MagGatedForCausalLM'
}
with open('$CKPT_PATH/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('  Added auto_map to config.json')
"
else
    echo "  auto_map already exists in config.json, skipping"
fi

echo ""
echo "Done! Files added to checkpoint:"
ls -la "$CKPT_PATH/configuration_mag_gated.py" "$CKPT_PATH/modeling_mag_gated.py"
echo ""
echo "config.json auto_map:"
python3 -c "
import json
with open('$CKPT_PATH/config.json') as f:
    c = json.load(f)
print(json.dumps(c.get('auto_map', {}), indent=2))
"
echo ""
echo "Now you can use:"
echo "  lm_eval --model hf --trust_remote_code --model_args pretrained=$CKPT_PATH ..."
