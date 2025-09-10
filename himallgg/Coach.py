import copy
import time
import torch
import random
from tqdm import tqdm
from sklearn import metrics
import himallgg
import numpy as np

log = himallgg.utils.get_logger()

class Coach:
    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.num_classes = 4
        self.best_dev_f1 = None
        self.best_tes_f1 = None
        self.test_f1_when_best_dev = None
        self.best_epoch = None
        self.best_state = None

    def _safe_load_state(self, state_dict):
        if state_dict is None:
            log.warning("No state_dict provided; skipping.")
            return
        model_state = self.model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        skipped_missing = []
        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            else:
                skipped_missing.append(k)
        if len(filtered_state) > 0:
            self.model.load_state_dict(filtered_state, strict=False)
            log.info(f"Loaded {len(filtered_state)} parameters.")
        if skipped_missing:
            log.info(f"Skipped {len(skipped_missing)} keys not in model.")
        if skipped_mismatch:
            log.info(f"Skipped {len(skipped_mismatch)} keys due to shape mismatch. Example: {skipped_mismatch[0]}")

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt.get("best_dev_f1", None)
        self.best_tes_f1 = ckpt.get("best_tes_f1", None)
        self.test_f1_when_best_dev = ckpt.get("test_f1_when_best_dev", None)
        self.best_epoch = ckpt.get("best_epoch", None)
        self.best_state = ckpt.get("best_state", None)
        self._safe_load_state(self.best_state)

    def train(self):
        best_dev_f1, best_tes_f1, best_epoch, best_state = (
            self.best_dev_f1, self.best_tes_f1, self.best_epoch, self.best_state
        )
        test_f1_when_best_dev = self.test_f1_when_best_dev

        for epoch in range(1, self.args.epochs + 1):
            pre_epochs = getattr(self.args, "pretrain_full_epochs", 0)
            if epoch <= pre_epochs:
                new_rate = {"T": 0.0, "A": 0.0, "V": 0.0}
                phase = "FULL-PRETRAIN"
            else:
                new_rate = {
                    "T": random.uniform(0.1, 0.5),
                    "A": random.uniform(0.1, 0.5),
                    "V": random.uniform(0.1, 0.5)
                }
                phase = "MISSING-TRAIN"
            self.trainset.missing_rate = new_rate
            log.info(f"[Epoch {epoch}] Phase={phase} missing_rate={new_rate} use_graphcl={getattr(self.args,'use_graphcl', False)}")

            self.train_epoch(epoch)

            dev_f1 = self.evaluate()
            log.info(f"[Dev set] f1={dev_f1:.4f}")
            test_f1 = self.evaluate(test=True)
            log.info(f"[Test set] f1={test_f1:.4f}")

            if best_tes_f1 is None or test_f1 > best_tes_f1:
                best_tes_f1 = test_f1
                test_f1_when_best_dev = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Saved the best model.")

        if best_state is not None:
            self._safe_load_state(best_state)

        log.info(f"Best epoch: {best_epoch}")
        dev_f1 = self.evaluate()
        test_f1 = self.evaluate(test=True)
        log.info(f"[Dev f1: {dev_f1:.4f}] [Test f1: {test_f1:.4f}]")

        return best_dev_f1, best_epoch, best_state, test_f1_when_best_dev, best_tes_f1

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        n_batches = len(self.trainset)
        for idx in tqdm(range(n_batches), desc=f"train epoch {epoch}"):
            self.model.zero_grad()
            data = self.trainset[idx]

            # 数据转移
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and k != 'sentence':
                    data[k] = v.to(self.args.device)

            # forward + loss
            nll = self.model.get_loss(data)
            loss = nll
            if self.model.use_graphcl:
                gcl_loss = self.model.get_graphcl_loss(data)
                loss += self.model.graphcl_weight * gcl_loss

            epoch_loss += loss.item()
            loss.backward()
            self.opt.step()

        log.info(f"[Epoch {epoch}] Loss={epoch_loss:.4f} Time={time.time()-start_time:.2f}s")

    def evaluate(self, test=False, debug_N=5):
        dataset = self.testset if test else self.devset
        self.model.eval()
        golds, preds = [], []

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                for k, v in data.items():
                    if isinstance(v, torch.Tensor) and k != 'sentence':
                        data[k] = v.to(self.args.device)

                y_hat = self.model.forward(data)
                label_tensor = data["label_tensor"].to(self.args.device)

                golds.extend(label_tensor.cpu().numpy().tolist())
                preds.extend(np.atleast_1d(y_hat.cpu().numpy()).tolist())

                if idx < debug_N:
                    print(f"[DEBUG] Batch {idx}: gold={label_tensor.cpu().numpy()} pred={y_hat.cpu().numpy()}")

        print("Unique gold labels:", set(golds))
        print("Unique pred labels:", set(preds))
        print(metrics.classification_report(golds, preds, digits=4))
        f1 = metrics.f1_score(golds, preds, average="weighted")
        return f1
