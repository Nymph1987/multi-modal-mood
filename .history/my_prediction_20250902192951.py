import argparse
import torch
import himallgg
from himallgg.Dataset import Dataset  # 自己写的 Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

log = himallgg.utils.get_logger()


def main(args):
    # 设置随机种子
    himallgg.utils.set_seed(args.seed)

    log.debug(f"Loading data from '{args.data}'.")
    data = himallgg.utils.load_pkl(args.data)
    log.info("Loaded data.")
    log.info(f"Data keys: {list(data.keys())}")

    # 初始化模型
    model = himallgg.LGGCN(args).to(args.device)

    # 加载 checkpoint（只加载模型权重）
    ckpt = torch.load("save/IEMOCAP/modeltav.pt", map_location=args.device)
    if "best_state" in ckpt:
        model.load_state_dict(ckpt["best_state"])
        log.info("Checkpoint loaded successfully.")
    else:
        log.warning("Checkpoint does not contain 'best_state', trying full ckpt as state_dict.")
        model.load_state_dict(ckpt)

    # 遍历缺失率
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for rate in missing_rates:
        log.info(f"==== Evaluating with missing rate {rate} ====")

        # 创建 Dataset，支持自定义缺失模态
        testset = Dataset(
            samples=data["test"],
            batch_size=args.batch_size,
            missing_rate={"text": 0.0, "audio": rate, "visual": rate},  # 例子
            missing_modalities=["text", "audio", "visual"],  # 指定可能缺失的模态
            seed=args.seed
        )

        # 初始化 Prediction
        pred = himallgg.Prediction(model, testset, device=args.device)

        # 评估
        test_f1 = pred.evaluate()
        log.info(f"[Test set] [Missing {rate:.1f}] [f1 {test_f1:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prediction.py")
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--class_weight", action="store_true", help="Use class weights in nll loss.")

    # Model parameters
    parser.add_argument("--drop_rate", type=float, default=0.4, help="Dropout rate.")
    parser.add_argument("--wp", type=int, default=10, help="Past context window size.")
    parser.add_argument("--wf", type=int, default=10, help="Future context window size.")
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of directions.")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn", type=str, default="lstm", choices=["lstm", "gru"], help="Type of RNN cell.")
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    parser.add_argument("--scale_para", type=float, default=-0.5, help="Attention scale para.")

    args = parser.parse_args()
    log.debug(args)
    main(args)
