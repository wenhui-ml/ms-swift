#!/usr/bin/env python3
"""
将 MagGated checkpoint 准备为可独立加载的格式。

问题：MagGated 是自定义架构，checkpoint 目录中不包含 Python 模型代码。
  lm_eval / swift eval 等外部工具用 AutoConfig.from_pretrained() 加载时，
  即使设了 --trust_remote_code，也找不到 model_type="mag_gated" 对应的类。

解决方案：生成一个自包含的 modeling_mag_gated.py（内嵌 Config 类），
  并在 config.json 中添加 auto_map 字段，使 HuggingFace AutoClasses
  能通过 trust_remote_code=True 自动加载。

Usage:
  python configs/prepare_checkpoint_for_eval.py <checkpoint_path>
  python configs/prepare_checkpoint_for_eval.py output_bm/mag_gated-d1024-L28/v0-20260327-071047/checkpoint-300

批量处理：
  python configs/prepare_checkpoint_for_eval.py output_bm/*/v0-*/checkpoint-*
  python configs/prepare_checkpoint_for_eval.py model_checkpoints/mag_gated-*
"""

import json
import sys
import os
from pathlib import Path


def prepare_checkpoint(ckpt_path: str):
    ckpt = Path(ckpt_path)
    config_file = ckpt / "config.json"

    if not config_file.exists():
        print(f"  SKIP: No config.json at {ckpt}")
        return False

    with open(config_file) as f:
        config = json.load(f)

    if config.get("model_type") != "mag_gated":
        print(f"  SKIP: model_type is '{config.get('model_type')}', not 'mag_gated'")
        return False

    # Find the source model code
    script_dir = Path(__file__).resolve().parent
    swift_root = script_dir.parent
    src_config = swift_root / "swift" / "model" / "mag_gated" / "configuration_mag_gated.py"
    src_modeling = swift_root / "swift" / "model" / "mag_gated" / "modeling_mag_gated.py"

    if not src_modeling.exists():
        print(f"  ERROR: Cannot find {src_modeling}")
        return False

    print(f"  Preparing: {ckpt}")

    # --- Step 1: Copy configuration_mag_gated.py as-is ---
    config_code = src_config.read_text()
    (ckpt / "configuration_mag_gated.py").write_text(config_code)

    # --- Step 2: Create modeling_mag_gated.py with fixed import ---
    # Read the original modeling file
    modeling_code = src_modeling.read_text()

    # Replace relative import with absolute import that HF dynamic modules can resolve.
    # HF's dynamic module system copies files to a cache dir and adds it to sys.path,
    # so "from .configuration_mag_gated import" won't work but the module file
    # being in the same directory means we need a special approach.
    #
    # The trick: HF check_imports() scans for import statements and tries to verify
    # they are installed packages. We use importlib to do a runtime import instead,
    # which bypasses the static check.
    modeling_code = modeling_code.replace(
        "from .configuration_mag_gated import MagGatedConfig",
        "# Auto-import: load config from same directory (compatible with HF dynamic modules)\n"
        "import importlib, pathlib as _pl\n"
        "_cfg_path = _pl.Path(__file__).parent / 'configuration_mag_gated.py'\n"
        "_spec = importlib.util.spec_from_file_location('configuration_mag_gated', _cfg_path)\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mod)\n"
        "MagGatedConfig = _mod.MagGatedConfig"
    )

    # Also handle the non-relative form (in case the sed already ran)
    modeling_code = modeling_code.replace(
        "from configuration_mag_gated import MagGatedConfig",
        "# Auto-import: load config from same directory (compatible with HF dynamic modules)\n"
        "import importlib, pathlib as _pl\n"
        "_cfg_path = _pl.Path(__file__).parent / 'configuration_mag_gated.py'\n"
        "_spec = importlib.util.spec_from_file_location('configuration_mag_gated', _cfg_path)\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mod)\n"
        "MagGatedConfig = _mod.MagGatedConfig"
    )

    (ckpt / "modeling_mag_gated.py").write_text(modeling_code)

    # --- Step 3: Add auto_map to config.json ---
    config["auto_map"] = {
        "AutoConfig": "configuration_mag_gated.MagGatedConfig",
        "AutoModelForCausalLM": "modeling_mag_gated.MagGatedForCausalLM",
    }
    # Also ensure use_cache=True for inference (training checkpoints may have False)
    if config.get("use_cache") is False:
        config["use_cache"] = True
        print("    Fixed use_cache: False → True")

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"    ✓ configuration_mag_gated.py copied")
    print(f"    ✓ modeling_mag_gated.py created (self-contained)")
    print(f"    ✓ config.json updated with auto_map")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python configs/prepare_checkpoint_for_eval.py <checkpoint_path> [...]")
        sys.exit(1)

    paths = sys.argv[1:]
    success = 0
    for path in paths:
        if prepare_checkpoint(path):
            success += 1

    print(f"\nDone: {success}/{len(paths)} checkpoints prepared.")
    if success > 0:
        print("\nYou can now run:")
        print("  lm_eval --model hf --trust_remote_code \\")
        print("    --model_args pretrained=<checkpoint_path> \\")
        print("    --tasks mmlu,arc_easy,hellaswag \\")
        print("    --batch_size auto --num_fewshot 0")


if __name__ == "__main__":
    main()
