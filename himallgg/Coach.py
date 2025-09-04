import copy
import time
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn import metrics
import himallgg

log = himallgg.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        self.best_dev_f1 = None
        self.best_tes_f1 = None
        self.test_f1_when_best_dev = None
        self.best_epoch = None
        self.best_state = None

    def _safe_load_state(self, state_dict):
        if state_dict is None:
            log.warning("No state_dict provided for safe load; skipping.")
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
        if len(filtered_state) == 0:
            log.warning("No matching parameters found in given state_dict. Model will use current initialization.")
        else:
            self.model.load_state_dict(filtered_state, strict=False)
            log.info(f"Loaded {len(filtered_state)} matching parameters from state_dict.")
        if skipped_missing:
            log.info(f"Skipped {len(skipped_missing)} keys not present in current model (e.g., new modules like DMC/Reconstruct).")
        if skipped_mismatch:
            log.info(f"Skipped {len(skipped_mismatch)} keys due to shape mismatch (checkpoint vs current). "
                     f"Example: {skipped_mismatch[0]}")

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_tes_f1 = ckpt["best_tes_f1"]
        self.test_f1_when_best_dev = ckpt['test_f1_when_best_dev']
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self._safe_load_state(self.best_state)

    def train(self):
        log.debug(self.model)
        best_dev_f1, best_tes_f1, best_epoch, best_state = (
            self.best_dev_f1, self.best_tes_f1, self.best_epoch, self.best_state
        )
        test_f1_when_best_dev = self.test_f1_when_best_dev

        # Train
        for epoch in range(1, self.args.epochs + 1):
            # ⭐ 每轮随机生成三个模态的缺失率 dict
            new_rate = {
                "T": random.uniform(0.1, 0.5),
                "A": random.uniform(0.1, 0.5),
                "V": random.uniform(0.1, 0.5)
            }
            self.trainset.missing_rate = new_rate
            log.info(f"[Epoch {epoch}] Using missing_rate={new_rate}")

            self.train_epoch(epoch)
            dev_f1 = self.evaluate()
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            test_f1 = self.evaluate(test=True)
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))

            if best_tes_f1 is None or test_f1 > best_tes_f1:
                best_tes_f1 = test_f1
                test_f1_when_best_dev = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

        if best_state is None:
            log.warning("No best_state from training; using current model parameters without reloading.")
        else:
            self._safe_load_state(best_state)

        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1 = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1 = self.evaluate(test=True)
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))

        return best_dev_f1, best_epoch, best_state, test_f1_when_best_dev, best_tes_f1

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                if k != 'sentence':
                    data[k] = v.to(self.args.device)
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
        self.model.eval()
        golds, preds = [], []
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                for k, v in data.items():
                    if k != 'sentence':
                        data[k] = v.to(self.args.device)

                y_hat = self.model(data)

                text_len_tensor = data["text_len_tensor"]
                label_tensor = data["label_tensor"]
                label_list = [label_tensor[i, :L].cpu() for i, L in enumerate(text_len_tensor)]
                gold_nodes = torch.cat(label_list, dim=0)

                golds.extend(gold_nodes.numpy().tolist())
                preds.extend(y_hat.view(-1).cpu().numpy().tolist())

        assert len(golds) == len(preds), f"Length mismatch: golds={len(golds)}, preds={len(preds)}"
        print(metrics.classification_report(golds, preds, digits=4))
        f1 = metrics.f1_score(golds, preds, average="weighted")
        return f1
