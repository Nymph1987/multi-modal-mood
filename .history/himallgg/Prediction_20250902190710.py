import torch
from tqdm import tqdm
from sklearn import metrics


class Prediction:

    def __init__(self, testset, model, args):
        self.testset = testset
        self.model = model
        self.args = args

    def evaluate(self):
        dataset = self.testset

        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test"):
                data = dataset[idx]
                golds.append(data["label_tensor"])

                # 把输入搬到 device 上
                for k, v in data.items():
                    if k == "sentence":
                        continue
                    else:
                        data[k] = v.to(self.args.device)

                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()

            print(metrics.classification_report(golds, preds, digits=4))
            f1 = metrics.f1_score(golds, preds, average="weighted")

        return f1
