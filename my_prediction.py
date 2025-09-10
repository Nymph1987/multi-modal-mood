import argparse
import torch
import himallgg
from himallgg.Dataset import Dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def parse_rates(rates_str: str):
    try:
        return [float(x) for x in rates_str.split(",") if x.strip() != ""]
    except Exception as e:
        raise ValueError(f"--eval_missing_rates 解析失败: {rates_str}") from e

def main(args):
    himallgg.utils.set_seed(args.seed)

    log.info(f"Loading data from '{args.data}' ...")
    data = himallgg.utils.load_pkl(args.data)
    log.info(f"Loaded. Splits={list(data.keys())}")

    # 构建模型（需与训练时结构一致：rnn 类型、是否启用 mamba、隐藏维度等）
    model = himallgg.LGGCN(args).to(args.device)

    # 加载 checkpoint
    log.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device)
    if isinstance(ckpt, dict) and "best_state" in ckpt:
        state_dict = ckpt["best_state"]
    else:
        # 兼容直接保存 state_dict 的情况
        state_dict = ckpt
    # 过滤不匹配键
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            filtered_state_dict[k] = v
        else:
            if k not in model_state_dict:
                log.warning(f"[CKPT] Unexpected key: {k}")
            else:
                log.warning(f"[CKPT] Shape mismatch: {k} {v.shape} != {model_state_dict[k].shape}")
    # 缺失键提示
    miss = [k for k in model_state_dict.keys() if k not in state_dict]
    if miss:
        log.warning(f"[CKPT] Missing {len(miss)} keys, loading in strict=False")
    model.load_state_dict(filtered_state_dict, strict=False)
    log.info("Checkpoint loaded.")

    # 缺失率列表
    eval_rates = parse_rates(args.eval_missing_rates)

    for rate in eval_rates:
        log.info(f"==== Evaluating with missing_rate={rate} on modalities={args.missing_modalities} ====")
        testset = Dataset(
            data["test"],
            batch_size=args.batch_size,
            missing_rate={"text": rate, "audio": rate, "vision": rate},
            missing_modalities=list(args.missing_modalities),
            device=args.device
        )
        pred = himallgg.Prediction(model, testset, device=args.device)
        f1 = pred.evaluate()
        log.info(f"[Test] missing={rate:.2f} weighted-F1={f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LGGCN checkpoint on IEMOCAP/MOSI")
    parser.add_argument("--data", type=str, required=True, help="Path to data pkl")
    parser.add_argument("--dataset", type=str, default="iemocap", choices=["iemocap", "mosi"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)

    # 模型结构参数（务必与训练一致）
    parser.add_argument("--rnn", type=str, default="gru", choices=["lstm", "gru", "mamba"])
    parser.add_argument("--n_speakers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--drop_rate", type=float, default=0.4)
    parser.add_argument("--wp", type=int, default=10)
    parser.add_argument("--wf", type=int, default=10)
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--scale_para", type=float, default=-0.5)

    # Mamba 相关（如训练时启用，评估也需对应启用）
    parser.add_argument("--use_mamba_modal", action="store_true")
    parser.add_argument("--use_mamba_global", action="store_true")
    parser.add_argument("--use_mamba_sgcn", action="store_true")
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=3)
    parser.add_argument("--mamba_expand", type=int, default=2)

    # 缺失视角控制
    parser.add_argument("--missing_modalities", type=str, default="tav",
                        help="子集于 'tav'，如 tav / av / tv / ta 等")
    parser.add_argument("--eval_missing_rates", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5",
                        help="多个缺失率逗号分隔，如 '0.0,0.3,0.5'")

    # checkpoint
    parser.add_argument("--ckpt", type=str, default="save/IEMOCAP/modeltav.pt",
                        help="Path to checkpoint (.pt)")

    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    main(args)
