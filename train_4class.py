import argparse
import os
import sys
import torch
import himallgg
import logging

# ------------------ 日志设置 ------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_4class_log.txt")

log = himallgg.utils.get_logger()
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, message):
        if message.strip() != "":
            self.logger.log(self.level, message.strip())
    def flush(self):
        pass

sys.stdout = LoggerWriter(log, logging.INFO)
sys.stderr = LoggerWriter(log, logging.WARNING)

# ------------------ GPU 检查 ------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("CUDA available:", torch.cuda.is_available())

# ------------------ 主函数 ------------------
def main(args):
    himallgg.utils.set_seed(args.seed)

    # 加载数据
    log.info(f"Loading data from {args.data} ...")
    data = himallgg.utils.load_pkl(args.data)
    log.info("Data loaded.")

    # 构建 Dataset
    def build_ds(split):
        return himallgg.Dataset(
            data[split],
            device=args.device,
            batch_size=args.batch_size,
            missing_rate=args.missing_rate,
            missing_modalities=list(args.missing_modalities)
        )

    trainset = build_ds("train")
    devset = build_ds("dev")
    testset = build_ds("test")

    # 构建模型
    log.info("Building 4-class LGGCN model...")
    model_file = "./save/IEMOCAP/model_4class_all.pt"
    model = himallgg.LGGCN(args).to(args.device)

    # 优化器
    opt = himallgg.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    # Coach
    coach = himallgg.Coach(trainset, devset, testset, model, opt, args)

    # 载入 checkpoint
    if not args.from_begin and os.path.exists(model_file):
        log.info(f"Loading checkpoint from {model_file} ...")
        ckpt = torch.load(model_file, map_location=args.device)
        coach.load_ckpt(ckpt)
    else:
        log.info("Training from scratch.")

    # 训练
    log.info("Start training 4-class model...")
    ret = coach.train()

    # 保存 checkpoint
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
        "test_f1_when_best_dev": ret[3],
        "best_tes_f1": ret[4],
    }
    torch.save(checkpoint, model_file)
    log.info(f"[4-CLASS TRAINING] Best model saved (epoch {ret[1]}) to {model_file}.")


# ------------------ 参数解析 ------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument("--dataset", type=str, default="iemocap", choices=["iemocap", "mosi"], help="Dataset name")

    # Training parameters
    parser.add_argument("--from_begin", action="store_true", help="Training from begin.")
    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "rmsprop", "adam"], help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float, help="Max grad norm")
    parser.add_argument("--drop_rate", type=float, default=0.4, help="Dropout rate.")

    # Model parameters
    parser.add_argument("--wp", type=int, default=10, help="Past context window size.")
    parser.add_argument("--wf", type=int, default=10, help="Future context window size.")
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn", type=str, default="gru", choices=["lstm", "gru", "mamba"], help="Type of RNN cell.")  
    parser.add_argument("--class_weight", action="store_true", help="Use class weights in nll loss.")

    # Missing-view on-the-fly 生成（Dataset用）
    parser.add_argument("--missing_rate", type=float, default=0.0, help="On-the-fly missing rate in Dataset")
    parser.add_argument("--missing_modalities", type=str, default="av", help="Which modalities can be missing, from 't','a','v'")

    # Integrity loss weights
    parser.add_argument("--integrity_alpha", type=float, default=1.0, help="Weight for reconstruction MSE")
    parser.add_argument("--integrity_beta", type=float, default=1.0, help="Weight for text consistency")
    parser.add_argument("--integrity_weight", type=float, default=1.0, help="Global weight for integrity loss sum")

    # GraphCL 对比学习（默认关闭，不影响现有训练）
    parser.add_argument("--use_graphcl", action="store_true", help="Enable GraphCL contrastive learning as graph reconstructor")
    parser.add_argument("--graphcl_weight", type=float, default=1.0, help="Weight for GraphCL contrastive loss")
    parser.add_argument("--graphcl_tau", type=float, default=0.5, help="Temperature for NT-Xent")
    parser.add_argument("--graphcl_proj_dim", type=int, default=64, help="Projection head output dimension")
    parser.add_argument("--graphcl_aug_type", type=str, default="mask_features", choices=["identity", "mask_features", "drop_edges"], help="Graph augmentation type for GraphCL")
    parser.add_argument("--graphcl_aug_ratio", type=float, default=0.2, help="Augmentation ratio for GraphCL")

    # 两阶段训练
    parser.add_argument("--pretrain_full_epochs", type=int, default=0, help="Number of epochs to pretrain with full modalities (no missing)")

    parser.add_argument("--use_mamba_modal", action="store_true")
    parser.add_argument("--use_mamba_global", action="store_true")
    parser.add_argument("--use_mamba_sgcn", action="store_true")
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=3)
    parser.add_argument("--mamba_expand", type=int, default=2)


    # Others
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    args = parser.parse_args()
    log.debug(args)
    main(args)
