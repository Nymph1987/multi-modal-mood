import torch
from tqdm import tqdm
from sklearn import metrics
import himallgg

log = himallgg.utils.get_logger()

class Prediction:
    def __init__(self, testset, model, args):
        self.testset = testset
        self.model = model
        self.args = args
        self.best_dev_f1 = None
        self.best_tes_f1 = None
        self.test_f1_when_best_dev = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt.get("best_dev_f1", None)
        self.best_tes_f1 = ckpt.get("best_tes_f1", None)
        self.test_f1_when_best_dev = ckpt.get("test_f1_when_best_dev", None)
        self.best_epoch = ckpt.get("best_epoch", None)
        self.best_state = ckpt.get("best_state", None)
        if self.best_state:
            self.model.load_state_dict(self.best_state)

    def evaluate(self):
        dataset = self.testset
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="test"):
                batch = dataset[idx]  # dict
                # 移动到 device
                for k, v in batch.items():
                    if k != "sentence" and isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.args.device)
                # 模型前向
                y_hat = self.model(batch)
                all_preds.append(y_hat.detach().cpu())
                all_labels.append(batch["label_tensor"].cpu())

        preds = torch.cat(all_preds, dim=-1).numpy()
        labels = torch.cat(all_labels, dim=-1).numpy()

        print(metrics.classification_report(labels, preds, digits=4))
        f1 = metrics.f1_score(labels, preds, average="weighted")
        return f1
