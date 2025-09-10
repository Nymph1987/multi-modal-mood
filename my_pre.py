# import argparse
# import torch
# import himallgg
# from himallgg.Dataset import Dataset
# from tqdm import tqdm
# import logging
# import os
# import sys


# class DatasetWithLabels(Dataset):
#     """继承 Dataset，只增加 'labels' 键供 Prediction.evaluate 使用"""
#     def __getitem__(self, index):
#         batch = super().__getitem__(index)
#         # 每条 sample 只保留一个标签
#         if "label_tensor" in batch:
#             batch["labels"] = batch["label_tensor"]
#         return batch

# def main(args):
#     himallgg.utils.set_seed(args.seed)

#     log.debug(f"Loading data from '{args.data}'.")
#     data = himallgg.utils.load_pkl(args.data)
#     log.info("Loaded data.")
#     log.info(f"Data keys: {list(data.keys())}")

#     # 初始化模型
#     model = himallgg.LGGCN(args).to(args.device)

#     # 加载 checkpoint
#     ckpt = torch.load("save/IEMOCAP/modeltav.pt", map_location=args.device)
#     print("Checkpoint keys:", ckpt.keys())
    
#     # 获取模型的状态字典和checkpoint的状态字典
#     model_state_dict = model.state_dict()
#     checkpoint_state_dict = ckpt["best_state"]
    
#     # 过滤掉不匹配的键值
#     filtered_state_dict = {}
#     for key, value in checkpoint_state_dict.items():
#         if key in model_state_dict:
#             # 检查形状是否匹配
#             if model_state_dict[key].shape == value.shape:
#                 filtered_state_dict[key] = value
#             else:
#                 log.warning(f"Shape mismatch for key '{key}': model {model_state_dict[key].shape} vs checkpoint {value.shape}")
#         else:
#             log.warning(f"Unexpected key in checkpoint: '{key}'")
    
#     # 检查是否有模型中的键值在checkpoint中缺失
#     for key in model_state_dict.keys():
#         if key not in checkpoint_state_dict:
#             log.warning(f"Missing key in checkpoint: '{key}'")
    
#     # 加载过滤后的状态字典
#     model.load_state_dict(filtered_state_dict, strict=False)
#     log.info("Checkpoint loaded successfully with filtered state dict.")

#         # 定义不同模态组合
#     modality_combinations = {
#         "text_only": {"text": 0.0, "audio": 1.0, "vision": 1.0},        # 只保留text
#         "audio_only": {"text": 1.0, "audio": 0.0, "vision": 1.0},       # 只保留audio  
#         "vision_only": {"text": 1.0, "audio": 1.0, "vision": 0.0},      # 只保留vision
#         "text+audio": {"text": 0.0, "audio": 0.0, "vision": 1.0},       # 保留text+audio
#         "text+vision": {"text": 0.0, "audio": 1.0, "vision": 0.0},      # 保留text+vision
#         "audio+vision": {"text": 1.0, "audio": 0.0, "vision": 0.0},     # 保留audio+vision
#         "all_modalities": {"text": 0.0, "audio": 0.0, "vision": 0.0}    # 全模态
#     }

#     for name, missing_rates in modality_combinations.items():
#         log.info(f"==== Evaluating with modalities: {name} ====")
    
#         testset = DatasetWithLabels(
#             data["test"],
#             batch_size=args.batch_size,
#             missing_rate=missing_rates,  # 修改这里
#             missing_modalities=["t", "a", "v"],  # 修改这里：所有模态都可以被缺失
#             device=args.device
#         )
    
#         pred = himallgg.Prediction(model, testset, device=args.device)
#         test_f1 = pred.evaluate()
#         log.info(f"[Test set] [{name}] [f1 {test_f1:.4f}]")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Test pre-trained LGGCN on IEMOCAP")
#     parser.add_argument("--data", type=str, required=True, help="Path to data")
#     parser.add_argument("--dataset", type=str, default="iemocap", choices=["iemocap", "mosi"], help="Dataset name")
#     parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
#     parser.add_argument("--class_weight", action="store_true", help="Use class weights in nll loss.")
#     parser.add_argument("--drop_rate", type=float, default=0.4)
#     parser.add_argument("--wp", type=int, default=10)
#     parser.add_argument("--wf", type=int, default=10)
#     parser.add_argument("--n_speakers", type=int, default=2)
#     parser.add_argument("--hidden_size", type=int, default=100)
#     parser.add_argument("--rnn", type=str, default="gru", choices=["lstm", "gru", "mamba"])
#     parser.add_argument("--seed", type=int, default=24)
#     parser.add_argument("--scale_para", type=float, default=-0.5)
    
#     # 添加缺失的Mamba相关参数
#     parser.add_argument("--use_mamba_modal", action="store_true", help="Use Mamba in modal encoders")
#     parser.add_argument("--use_mamba_global", action="store_true", help="Use Mamba in global graph")
#     parser.add_argument("--use_mamba_sgcn", action="store_true", help="Use Mamba in local graph SGCNs")
#     parser.add_argument("--mamba_layers", type=int, default=2, help="Number of Mamba layers")
#     parser.add_argument("--mamba_d_state", type=int, default=16, help="Mamba d_state parameter")
#     parser.add_argument("--mamba_d_conv", type=int, default=3, help="Mamba d_conv parameter")
#     parser.add_argument("--mamba_expand", type=int, default=2, help="Mamba expand parameter")
    
#     # 添加完整性约束相关参数
#     parser.add_argument("--integrity_alpha", type=float, default=1.0, help="Weight for reconstruction MSE")
#     parser.add_argument("--integrity_beta", type=float, default=1.0, help="Weight for text consistency")
#     parser.add_argument("--integrity_weight", type=float, default=1.0, help="Global weight for integrity loss sum")
    
#     # 添加缺失模态相关参数
#     parser.add_argument("--missing_rate", type=float, default=0.0, help="On-the-fly missing rate in Dataset")
#     parser.add_argument("--missing_modalities", type=str, default="av", help="Which modalities can be missing, from 't','a','v'")

#     args = parser.parse_args()
#     log.debug(args)

#     main(args)



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

    modality_combinations = {
        "text_only": {"text": 0.0, "audio": 1.0, "vision": 1.0},        # 只保留text
        "audio_only": {"text": 1.0, "audio": 0.0, "vision": 1.0},       # 只保留audio  
        "vision_only": {"text": 1.0, "audio": 1.0, "vision": 0.0},      # 只保留vision
        "text+audio": {"text": 0.0, "audio": 0.0, "vision": 1.0},       # 保留text+audio
        "text+vision": {"text": 0.0, "audio": 1.0, "vision": 0.0},      # 保留text+vision
        "audio+vision": {"text": 1.0, "audio": 0.0, "vision": 0.0},     # 保留audio+vision
        "all_modalities": {"text": 0.0, "audio": 0.0, "vision": 0.0}    # 全模态
    }

    for name, missing_rates in modality_combinations.items():
        log.info(f"==== Evaluating with modalities: {name} ====")
    
        testset = Dataset(
            data["test"],
            batch_size=args.batch_size,
            missing_rate=missing_rates,  # 修改这里
            missing_modalities=["t", "a", "v"],  # 修改这里：所有模态都可以被缺失
            device=args.device
        )
    
        pred = himallgg.Prediction(model, testset, device=args.device)
        test_f1 = pred.evaluate()
        log.info(f"[Test set] [{name}] [f1 {test_f1:.4f}]")

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

    # checkpoint
    parser.add_argument("--ckpt", type=str, default="save/IEMOCAP/modeltav.pt",
                        help="Path to checkpoint (.pt)")

    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    main(args)
