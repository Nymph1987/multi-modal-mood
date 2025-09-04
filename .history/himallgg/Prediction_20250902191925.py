import torch
from sklearn.metrics import f1_score, classification_report


class Prediction:
    def __init__(self, model, dataset, device="cuda"):
        """
        model: 你的 LGGCN 模型
        dataset: 自定义 Dataset 对象（支持 missing_rate 和 missing_modalities）
        device: 'cuda' 或 'cpu'
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for idx in range(len(self.dataset)):
                batch = self.dataset[idx]

                # 将 batch 中的张量移动到设备
                text = batch["text_tensor"].to(self.device)
                audio = batch["audio_tensor"].to(self.device)
                visual = batch["visual_tensor"].to(self.device)
                speaker = batch["speaker_tensor"].to(self.device)
                label = batch["label_tensor"].to(self.device)

                # 调用模型
                outputs = self.model(text, audio, visual, speaker)
                preds = torch.argmax(outputs, dim=-1)

                y_true.extend(label.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        # 计算 F1
        f1 = f1_score(y_true, y_pred, average="weighted")
        print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
        return f1
