# Himal 多模态情感识别

一个基于浏览器的情感识别原型，将人脸表情、语音语调和转写文本放在同一页面对照，并给出融合结果。项目用于学习和演示，不用于心理诊断。

## 1. 目标与依据

面向学习情感识别的同学和原型开发者，主要解决多种模型输出分散、输入异常和模型故障难以区分的问题。开发过程中实际遇到了麦克风输入、繁体转写、中文情绪一直为 neutral 等问题，因此补充了设备选择、简体转换和明确的错误提示。

把不同模态放在一起能否帮助用户判断结果，是目前的产品假设，尚未通过用户访谈验证，也没有情绪准确率评测。

## 2. 使用流程

打开页面并授权摄像头和麦克风，选择输入设备后启动分析，也可以上传视频。

- **前端**采集画面与音频，通过 WebSocket 发送，展示转写、各模态情绪和融合结果。
- **服务端**使用 FastAPI 调度模型：YOLOv8-Face 和 Mini-Xception 处理表情，SpeechBrain 处理声学情绪，Silero VAD 分段后由 faster-whisper 转写，再进行文本分析。
- **数据**主要保存在会话内存中；上传视频存放在 `uploads/`，日志可能包含转写文字。目前没有用户数据库，录音录像需取得同意，上传文件和日志需自行管理。

## 3. 设计与取舍

- 先检测语音片段再转写，避免逐个音频小包识别。转写与情绪分析分开处理，情绪模型失败不应影响文字展示。
- 中文暂用关键词、否定和程度词规则，OpenCC 负责繁转简。规则在页面明确标为“非模型”，不把支持度包装成准确率；尚未接入中文神经情绪模型。
- 融合时统一标签，根据概率分布的熵动态分配权重，缺失模态不参与。声学权重分数乘以 0.25，以减少不可靠语调判断的影响；这不是实验确定的最优值，只有声音可用时其权重仍为 1。
- 复用预训练模型，没有重新训练统一网络。优先保留安全加载检查，不通过关闭检查来加载不兼容的旧权重。

## 4. 来源与贡献

从已有的 Web 页面、FastAPI 服务和三模态融合工程开始，本次主要完善了语音分段与错误提示、麦克风选择、简体转换、中文规则、融合结果展示和声音降权，没有自行训练模型。

主要使用的开源项目如下，第三方代码及权重的使用需遵守各自许可：

| 来源 | 用途 |
| --- | --- |
| [Lindevs YOLOv8-Face](https://github.com/lindevs/yolov8-face)、[face_classification](https://github.com/oarriaga/face_classification) | 人脸定位与 Mini-Xception 表情分类。 |
| [SpeechBrain IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)、[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) | 声学情绪分类及基础语音特征。 |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[CTranslate2](https://github.com/OpenNMT/CTranslate2)、[Silero VAD](https://github.com/snakers4/silero-vad) | 本地转写、推理与语音分段。 |
| [Transformers](https://github.com/huggingface/transformers)、[TextBlob](https://github.com/sloria/TextBlob)、[OpenCC](https://github.com/yichen0831/opencc-python) | 模型加载、英文文本回退及繁简转换。 |

## 5. AI 的使用

1. **采用了先记录链路再调整参数的建议。** 排查音频接收、分段、队列和页面回传，没有盲目降低 VAD 阈值。模拟两句回传能够更新页面，但连续真人输入仍需复测；临时调试文件已清理。
2. **修改了文本分析方案。** 英文模型和 TextBlob 不适合中文，改用明确标注的规则基线，并加入 OpenCC。情绪样例验证通过，实际使用也已确认能显示简体，但不代表中文语义识别已完成。
3. **采用了修正融合显示和降低声音影响的建议。** 主卡片改为显示融合结果，声音权重分数降至原来的四分之一。目前只完成代码核对，尚未验证新权重的联合效果。

## 6. 完成情况

“已验证”仅表示下表所述范围，不代表模型准确率已通过评测。

| 功能 | 状态 | 验证情况 |
| --- | --- | --- |
| 真人转写与简体显示 | 已验证 | 实际使用已确认能显示转录文字和简体中文，未测字错率及长期稳定性。 |
| 中文规则情绪 | 已验证 | “我很开心”“我很难过”“我很生气”等样例得到对应结果，复杂语义未评测。 |
| 连续转写与尾段去重 | 已实现未验证 | 模拟连续回传通过，真人连续下一句尚未确认；尾段去重测试曾失败。 |
| 设备选择、纯音频路径与启动提示 | 已实现未验证 | 已有实现及部分模拟检查，仍需完整浏览器现场验证。 |
| 人脸、声学及英文文本模型 | 已实现未验证 | 已核对调用链，尚未完成各模型的真实输入推理验证。 |
| 融合展示与声音降权 | 已实现未验证 | 代码已修改，新权重及三模态联合效果未验证。 |
| 视频上传与分析 | 已实现未验证 | 已有接口，视频格式和完整推理流程未验收。 |
| 新环境安装与 GPU 部署 | 已实现未验证 | 已整理依赖，未完成空环境安装验证。 |
| 中文神经情绪模型、性能及准确率评测 | 未实现 | 当前中文使用规则，尚无校准或带标注数据评测。 |

**已知问题：**规则难以理解反讽和复杂上下文；IEMOCAP 声学模型不保证适用于中文自然会话；连续转写和尾段去重仍需检查。分数不等于准确率，处理跟不上输入时队列可能积压。目前没有完善的访问控制和数据保留策略。

## 运行方法

建议使用独立 Python 3.9 环境。Torch 与 Torchaudio 必须同版本、同 CPU/CUDA 渠道；GPU 版本按 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/) 安装。现有环境为 Torch 2.4，旧 Wav2Vec2 `.bin` 加载受安全限制，需要可信的完整 safetensors 模型，或独立验证 Torch/Torchaudio ≥2.6 的兼容性。文本分类只接受 safetensors。

```bash
python -m pip install -r requirements.txt
python -m pip check
```

模型权重不随代码提交。当前默认模型位于项目兄弟目录 `../model/`，部分代码使用本机绝对路径，迁移时需调整：

| 模块 | 需要准备的内容 |
| --- | --- |
| 人脸与表情 | `yolov8-face/weights/yolov8n-face-lindevs.onnx` 和 Mini-Xception HDF5 权重。 |
| 声学情绪 | `wav2vec2-base/` 中的 SpeechBrain 配置、接口、分类 checkpoint 和标签编码器；完整基础模型可用 `LOCAL_WAV2VEC2_DIR` 指定。 |
| 转写 | 完整 CTranslate2 Whisper 目录，包含 `model.bin`、配置和 tokenizer，由 `LOCAL_WHISPER_DIR` 指定。 |
| 文本分类（可选） | 完整模型、tokenizer 和 safetensors，由 `TEXT_EMOTION_MODEL_DIR` 指定；语言通过 `TEXT_EMOTION_MODEL_LANGUAGE` 设置，不能将英文模型直接当作中文模型。 |

当前机器启动命令：

```bash
export LOCAL_WHISPER_DIR="$PWD/faster-whisper-large-v3"
export ASR_LOCAL_FILES_ONLY=1
export ASR_LANGUAGE=zh
python app.py
```

新环境使用自己的 `python app.py`。打开 `http://localhost:8000`，授权后选择真实麦克风。通过 `/startup_status` 查看模型错误，Whisper 会在首次转写时加载。
请注意，远程采集通常需要 HTTPS；当前服务不应直接暴露到公网。
