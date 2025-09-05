import torch
from himallgg.model.LGGCN import LGGCN
import argparse
import himallgg

log = himallgg.utils.get_logger()

def main(args):
    # 加载 checkpoint
    ckpt = torch.load(args.model_path, map_location='cpu')
    state_dict = ckpt.get('best_state', ckpt)  # 支持两种格式
    print(f"Checkpoint loaded from '{args.model_path}'")
    print(f"Top-level keys in checkpoint: {list(ckpt.keys())}")
    print(f"Number of parameters in checkpoint: {len(state_dict)}\n")

    # 初始化模型
    model = LGGCN(args)  # LGGCN 内部会用 args.device
    model_state_dict = model.state_dict()
    print(f"Number of parameters in model: {len(model_state_dict)}\n")

    # 对比 checkpoint 和模型
    missing_in_checkpoint = []
    extra_in_checkpoint = []
    shape_mismatch = []

    for key, value in state_dict.items():
        if key not in model_state_dict:
            extra_in_checkpoint.append(key)
        else:
            if value.shape != model_state_dict[key].shape:
                shape_mismatch.append((key, value.shape, model_state_dict[key].shape))

    for key in model_state_dict:
        if key not in state_dict:
            missing_in_checkpoint.append(key)

    # 输出结果
    print("==== Checkpoint keys not in model ====")
    for k in extra_in_checkpoint:
        print(k)
    print("\n==== Model keys missing in checkpoint ====")
    for k in missing_in_checkpoint:
        print(k)
    print("\n==== Shape mismatches ====")
    for k, ckpt_shape, model_shape in shape_mismatch:
        print(f"{k}: checkpoint {ckpt_shape} vs model {model_shape}")

    print("\n==== Summary ====")
    print(f"Total checkpoint params: {len(state_dict)}")
    print(f"Total model params: {len(model_state_dict)}")
    print(f"Extra in checkpoint: {len(extra_in_checkpoint)}")
    print(f"Missing in checkpoint: {len(missing_in_checkpoint)}")
    print(f"Shape mismatches: {len(shape_mismatch)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check checkpoint and model parameter compatibility")
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for model")
    
    # 模型参数（必须和训练时一致，否则可能 shape 不匹配）
    parser.add_argument("--rnn", type=str, default="mamba", choices=["lstm", "gru", "mamba"])
    parser.add_argument("--n_speakers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--drop_rate", type=float, default=0.4)
    parser.add_argument("--wp", type=int, default=10)
    parser.add_argument("--wf", type=int, default=10)
    parser.add_argument("--scale_para", type=float, default=-0.5)
    parser.add_argument("--use_mamba_modal", action="store_true")
    parser.add_argument("--use_mamba_global", action="store_true")
    parser.add_argument("--use_mamba_sgcn", action="store_true")
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=3)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--data", type=str, required=True, help="Path to data")

    args = parser.parse_args()
    main(args)
