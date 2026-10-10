# Himal 多模态情感识别 Web 原型

这是一个在已有工程上持续完善的情感识别原型。浏览器采集画面和声音，后端分别处理人脸表情、声学情绪和转写文本，再把可用结果合并展示。它适合演示、学习和对照不同模态的输出，不适合据此判断一个人的真实心理状态，也不能用于医疗诊断或自动作出重要决策。

## 1. 面向谁，解决什么问题

- **目标用户是需要观察多模态推理过程的学习者和原型开发者。** 单看表情、语调或文字容易得到不同结论；本项目把三条处理链放在同一页面，便于发现差异和检查模型是否真正工作。这是产品定位假设，不代表已经完成用户访谈或证明存在稳定的市场需求。
- **当前重点是让输入、结果和故障之间的关系更清楚。** 本轮对话中出现了真人语音转写、繁简显示、连续下一句不更新、中文被英文情绪模型处理等具体问题。用户已确认真人转写和简体显示成功，这支持继续完善交互与可观测性，但不能据此推导情绪识别准确率或用户满意度。
- **预期价值是减少接入和排查成本，而不是承诺更准确地“读懂情绪”。** 在同一入口查看各模态结果、来源和融合权重，有助于区分没有输入、模型未加载和推理结果不可靠。尚未进行正式的用户研究、性能基准或带标注数据的准确率评估。

## 2. 核心流程与前后端职责

1. **前端负责采集与展示。** `templates/index.html` 包含主要页面及交互脚本：申请摄像头、麦克风权限，枚举并选择麦克风，把画面和音频发送到 WebSocket，展示转写、单模态结果、来源提示和融合主卡片。音频请求 16 kHz，后端仍会按实际采样率重采样。若默认输入是 FF-Camera 等虚拟设备，应在麦克风选择框中改选真实设备；不能仅凭浏览器授权就认定收到了真人声音。
2. **后端负责模型生命周期、分段和推理调度。** `app.py` 使用 FastAPI；启动时后台加载模型，`/startup_status` 返回加载进度和分模型错误。实时入口为 `/ws_realtime`，另有 `/ws`、`/ws_server_video` 及上传入口 `/upload_video`。实时音频先经过连接独立的 Silero VAD 分段，再排队送入 faster-whisper，转写后做文本情绪分析；声学情绪使用独立短环形缓冲，不直接充当转写缓冲。
3. **各模型负责不同的信息来源。** `core/vision_pipeline.py` 串联 YOLOv8-Face 人脸定位与 Mini-Xception 表情分类；`core/audio_emotion_model.py` 调用 SpeechBrain Wav2Vec2-IEMOCAP 分类器；`core/text_emotion_model.py` 完成转写、OpenCC 简体转换及文本分类。中文默认走规则基线，英文在本地 Transformers 模型可用且语言匹配时使用模型，否则走 TextBlob 与关键词回退。
4. **融合只处理实际可用的分布。** `core/fusion_model.py` 将标签映射到七类情绪，按分布熵调整权重，再返回综合结果、参与模态和实际权重。模型不可用或没有有效输入不应被当作真实的 neutral 预测。没有可用分布时，融合返回的 neutral、0 分只是占位，不是识别结论。
5. **数据目前以会话状态和本地文件为主。** 实时缓冲、队列及最近结果保存在进程内；上传视频保存在项目的 `uploads/`，当前没有用户数据库或完整的数据保留策略。ASR 日志可能含转写原文；临时网络调试插桩及调试配置已按用户要求清理，业务日志仍保留。真实使用前应取得录音、录像同意，限制服务访问，并制定日志和上传文件的保留、删除规则；不要把当前原型当作已完成隐私治理的服务。

## 3. 设计取舍与已知限制

- **先分段再转写，避免把每个音频小包当成一句话。** 当前 VAD 默认阈值为 0.5，最短语音 250 ms，结束静音 500 ms，最长片段 15 s，强制切段重叠 250 ms。实时 ASR 队列采用不主动丢弃待处理片段的策略，优点是便于追踪遗漏，代价是处理跟不上输入时延迟和内存会增长，尚无生产级背压方案。
- **转写与文本情绪结果分开保留。** 即使文本情绪推理失败，已经识别的文字也不应丢失。OpenCC 只负责繁转简，不提升识别准确率，也不是情绪模型。英文 Twitter RoBERTa 不能直接当作中文分类器；中文规则明确返回 `chinese-rules`、`zh-rules-baseline-v1` 和未校准分数标记。
- **规则是可解释的临时基线，不是中文神经模型的替代品。** 中文实现覆盖关键词、部分否定和程度词；无命中时偏向 neutral，不能证明说话者真的没有情绪。反讽、复杂否定、上下文和混合情绪仍可能误判。界面的规则支持度和模型分数都不等于准确率，尚未进行概率校准。
- **采用决策级融合，便于接入缺失模态。** 不需要重新训练统一网络，但不同模型的标签、语言域和分数尺度并不一致，低熵也不等于可靠。声学情绪来自 IEMOCAP，当前映射四类 neutral/happy/sad/angry，不能视为中文自然会话的七类通用模型。归一化前将 audio 权重分数乘以 0.25 是保守的工程调整，不是实验得到的最优参数；只有 audio 可用时，归一化后的权重仍为 1。
- **视觉和语音采用不同运行后端。** YOLO ONNX 由 OpenCV DNN 执行，普通 `opencv-python` wheel 不保证支持 CUDA，代码可回退 CPU；Mini-Xception 固定在 TensorFlow CPU 上运行。Torch 和 faster-whisper 可使用 GPU，但它们对 CUDA/cuDNN 的要求需要分别检查，安装了 GPU Torch 并不意味着所有模块都能加速。
- **优先保留安全加载边界。** 当前 Torch 2.4 环境不能走代码中的 Wav2Vec2 旧 `pytorch_model.bin` 加载分支；需要同一 revision 的完整 safetensors 目录，或在独立环境中升级并验证匹配的 Torch/Torchaudio 2.6。文本 RoBERTa 当前只接受 safetensors。不要取消安全检查、拼接不同 revision 的配置，或加载来源不明的 checkpoint 来绕过报错。

## 4. 来源与本次贡献范围

原工程已经包含 Web 页面、FastAPI 服务、视觉/音频/文本模块及融合结构。本轮工作是在此基础上排查和完善，不把现有架构、第三方模型或预训练权重宣称为本次原创，也没有自行训练新的情感识别模型。

| 来源 | 在本项目中的用途与确认依据 |
| --- | --- |
| [Lindevs YOLOv8-Face](https://github.com/lindevs/yolov8-face) | `third_party/yolov8-face/README.md` 和检测器下载地址均指向 Lindevs；运行时使用 `yolov8n-face-lindevs.onnx` 做人脸检测，不做表情分类。 |
| [oarriaga/face_classification](https://github.com/oarriaga/face_classification) | 仓库内保留了第三方代码和说明，视觉模块加载其 FER2013 Mini-Xception HDF5 权重。上游报告的指标不代表本项目现场表现。 |
| [SpeechBrain IEMOCAP 模型](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) 与 [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | 本地 `hyperparams.yaml` 明确写出这两个来源，`custom_interface.py` 使用 SpeechBrain 接口。基础模型与 IEMOCAP 分类权重是两套不同的文件。 |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[CTranslate2](https://github.com/OpenNMT/CTranslate2) 与 [Silero VAD](https://github.com/snakers4/silero-vad) | 代码分别调用 `WhisperModel` 和 `load_silero_vad`；转写使用 CTranslate2 格式 Whisper 权重，VAD 使用安装包附带的模型。 |
| [Transformers](https://github.com/huggingface/transformers) | 提供 Wav2Vec2 和本地文本分类加载接口。文本目录名是 `twitter-roberta-base-emotion`，融合注释提及 CardiffNLP；仅凭目录名不宣称该本地副本的具体上游 revision 已核验。 |
| [TextBlob](https://github.com/sloria/TextBlob) 与 [OpenCC Python 实现](https://github.com/yichen0831/opencc-python) | 代码直接调用 TextBlob 的英文极性分析和 `OpenCC('t2s')`。中文规则及来源提示属于工程中的基线实现，不是上述库提供的中文神经情绪能力。 |

本轮完善集中在转写链路的可观测性、模型不可用时的错误呈现、本地模型安全加载、中文规则与来源透明、简体显示、融合主卡片及音频降权，以及这份说明和依赖清单。各项是否验证见下节，不能把“代码已经写入”与“真实模型已运行成功”混为一谈。分发第三方代码或权重前，仍需核对各上游许可证和使用条款。

### 本轮对 AI 建议的实际处理

1. **修改了“直接降低 VAD 阈值”的排查方向。** 没有把阈值盲目调低，而是先记录收到的音频、VAD 概率、分段、入队、转写完成与回传情况，以区分虚拟麦克风、未形成片段和队列问题。默认阈值仍为 0.5；连续下一句的真人复测尚未确认。用户已要求终止调试并清理临时插桩、日志与配置，这不代表问题已确认解决。
2. **否决了让英文回退承担中文语义识别的做法。** 中文改用透明的规则基线，保留模型、来源、标签和未校准分数说明；显示前通过 OpenCC 统一简体。规则已做样例验证，用户也确认简体显示成功，但中文神经情绪模型并未实现，不能宣传为已解决中文情绪理解。
3. **采用了融合主卡片与声学降权建议，但保留验证边界。** 主卡片改读融合结果，audio 分数在归一化前乘以 0.25，单模态结果继续用于对照。这些改动已有代码和静态检查依据；本次没有运行新权重的验证，更没有真实三模态联合准确率数据。

## 5. 完成情况与验证边界

状态统一使用 **未实现**、**已实现未验证**、**已验证**。其中“已验证”仅指表格所列方式和范围，不表示整个模块或模型质量已经验收。下表记录本轮对话已有反馈和代码核对结果；本次文档整理未重新执行推理。

| 主要功能 | 状态 | 验证方式、结果及尚存边界 |
| --- | --- | --- |
| 真人语音转写与文字展示 | 已验证 | 用户在本轮对话中确认真人转写成功；只证明已反馈的输入可以转写，未测字错率、长时间稳定性或多说话人。 |
| 转写与文本的简体显示 | 已验证 | 用户确认简体显示成功；代码使用 OpenCC `t2s`，不代表 ASR 准确率提升。 |
| 中文规则文本情绪 | 已验证 | 本轮此前已做规则样例验证，来源和未校准标记可从代码确认；不是现场中文神经模型推理，也不等于带标注数据评测通过。 |
| VAD 分段、连续下一句更新与尾段去重 | 已实现未验证 | 已有分段、队列与回传实现，连续下一句真人结果仍未确认；调试已按用户要求终止并清理，问题尚未确认解决。既有 `test_no_duplicate_overlap_tail` 曾失败，尚无修复后通过的证据；不能宣称测试套件全绿。 |
| 麦克风选择及摄像头不可用时的纯音频路径 | 已实现未验证 | 页面已有设备枚举、精确 deviceId 和纯音频回退；既有 mock/静态检查只覆盖控制逻辑，FF-Camera 与真实设备切换仍需浏览器现场确认。 |
| YOLO 人脸检测与 Mini-Xception 表情推理 | 已实现未验证 | 已核对代码、权重路径及 CPU/CUDA 回退逻辑；本轮执行环境曾报 `trae-sandbox not found`，没有真实人脸推理验证。 |
| SpeechBrain 声学情绪、标签解码与安全加载 | 已实现未验证 | 已有本地文件检查及按 label_encoder 解码的实现；没有真实声音神经推理验证。Torch 2.4 与旧 bin 权重的安全限制仍需处理。 |
| 英文本地 Transformers 文本情绪模型 | 已实现未验证 | 已有 safetensors、语义标签与语言匹配检查；本地权重来源和实际英文推理效果仍需独立核验，不能用于证明中文分类有效。 |
| 中文神经情绪模型 | 未实现 | 目前中文走规则基线。可配置模型目录的接口不等于已经部署、验证了适合中文的模型。 |
| 融合主卡片与 audio ×0.25 降权 | 已实现未验证 | 模板主卡片已读 `data.fusion`，融合代码已修改；只有静态/代码核对，新降权未运行验证，未做真人三模态联合验收。 |
| 视频上传及服务端视频多模态分析 | 已实现未验证 | 已有上传、解码和 WebSocket 路径；本次未验证不同容器、音轨、摄像头设备与真实推理结果。 |
| 启动进度、分模型错误和降级提示 | 已实现未验证 | `/startup_status` 和模型错误记录已存在，已有 mock 检查不能替代真实模型启动与异常恢复验收。 |
| 安装依赖与 GPU 部署指导 | 已实现未验证 | 已整理清单与安装步骤，但没有完成全新环境安装及 GPU 兼容性验证，不能宣称可一键部署。 |
| 性能基准、概率校准和情绪准确率评估 | 未实现 | 尚未建立并完成吞吐/延迟基准、概率校准或带标注数据的准确率评估。 |

后续应先复现两句连续真人输入，核对每句的分段、入队、解码和页面回传；再分别验证视觉与声学模型，最后对融合权重做同输入对照。`core/test_asr.py` 可用于回归分段和接口控制逻辑，但其中包含模拟 VAD、mock 及静态检查，不能用其结果代替神经模型质量评测。

## 6. 环境、模型准备与启动

### 依赖安装

当前可读取的环境位于 `/mnt/F/emotional/miniconda3/envs/himalgg`，使用 Python 3.9。已从安装包元数据确认 Torch/Torchaudio `2.4.0+cu121`、Transformers `4.57.1`、TensorFlow `2.20.0`、Keras `3.10.0`、SpeechBrain `1.0.3`、faster-whisper `1.2.1`、CTranslate2 `4.8.2`、Silero VAD `6.0.0`、NumPy `1.26.4` 和 OpenCV `4.8.0.76`。这些是安装记录，不是全链路运行成功的证明。

`requirements.txt` 按运行路径及必要依赖整理，保留 Python 3.9 可用的版本范围，不是整个 Conda 环境的 `pip freeze`，也不是已经验收的锁文件。YOLO 直接使用 OpenCV DNN，因此不单独要求 `onnxruntime-gpu` 或 Ultralytics；Silero/faster-whisper 所需的间接依赖由 pip 解析。TensorFlow 使用 `tf.keras` 加载 HDF5，不需要第三方训练脚本中的整套旧依赖，也不要求安装 `openai-whisper`。

建议先在独立 Python 3.9 环境安装**相同发行版本、相同 CPU/CUDA 渠道**的 Torch 和 Torchaudio。GPU wheel 应按驱动和平台从 [PyTorch 官方安装渠道](https://pytorch.org/get-started/locally/) 或 [历史版本说明](https://pytorch.org/get-started/previous-versions/) 选择；不要直接照搬本机的 `+cu121` 后缀。requirements 的范围不能代替版本配对，Torch/Torchaudio 升级后要分别检查 SpeechBrain 和本地权重加载。

```bash
cd /mnt/F/emotional/himal-my-model/himalgg_web_1
python -m pip install -r requirements.txt
python -m pip check
```

上面是待执行的安装步骤，本次没有验证全新安装成功。已有 GPU 环境应先检查解析计划，避免无意替换已配对的 Torch。faster-whisper 使用的 CTranslate2 还需要匹配的 CUDA/cuDNN 运行库，具体要求以对应版本的上游说明为准；TensorFlow 不需要额外安装 GPU 扩展，本项目将 Mini-Xception 放在 CPU。

### 模型路径

下列默认路径来自现有代码，不代表这些文件均已完成加载验证。`model/` 是项目的兄弟目录，迁移机器时需准备权重并处理默认绝对路径。

| 模块 | 默认位置与必要内容 |
| --- | --- |
| 人脸检测 | `/mnt/F/emotional/himal-my-model/model/yolov8-face/weights/yolov8n-face-lindevs.onnx`。缺失时检测器可能从上游下载；设 `MODEL_OFFLINE=1` 可禁止该下载。 |
| 表情分类 | `/mnt/F/emotional/himal-my-model/model/face_classification/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5`。 |
| 声学情绪分类 | `/mnt/F/emotional/himal-my-model/model/wav2vec2-base/`，需有 `custom_interface.py`、`hyperparams.yaml`、`wav2vec2.ckpt`、`model.ckpt` 和 `label_encoder.txt`。 |
| Wav2Vec2 基础模型 | 可用 `LOCAL_WAV2VEC2_DIR` 指向完整目录；默认查找 `/mnt/F/emotional/himal-my-model/model/facebook/wav2vec2-base` 及项目内 `wav2vec2_checkpoints/.../snapshots/`。同一目录需包含 `config.json`、`preprocessor_config.json` 和安全权重；仅旧 bin 时要求 Torch ≥2.6。 |
| Whisper 转写 | 用 `LOCAL_WHISPER_DIR` 指定完整 CTranslate2 模型目录。本项目已有 `faster-whisper-large-v3/` 的配置文件；启动前仍需确认 `model.bin`、`config.json`、`tokenizer.json` 等文件齐全。Transformers 的 Wav2Vec2 权重不能用于 Whisper。 |
| 本地文本分类 | 默认 `/mnt/F/emotional/himal-my-model/model/twitter-roberta-base-emotion`；可用 `TEXT_EMOTION_MODEL_DIR` 覆盖。需完整模型、tokenizer 和 safetensors，`config.id2label` 必须是真实语义标签。`TEXT_EMOTION_MODEL_LANGUAGE` 默认为 `en`，不能仅改成 `zh` 就把英文模型变成中文模型。 |

如果只有旧 Wav2Vec2 bin，先在独立环境安装匹配的 Torch/Torchaudio 2.6，再验证 SpeechBrain、Transformers、CUDA 和 checkpoint 兼容性，验证成功后才迁移。另一条路径是部署可信来源、同一 revision 的完整 safetensors 基础模型；该选择也不代表其他 `.ckpt` 文件已经完成安全和兼容验证。

### 启动与检查

```bash
cd /mnt/F/emotional/himal-my-model/himalgg_web_1
export LOCAL_WHISPER_DIR="$PWD/faster-whisper-large-v3"
export ASR_LOCAL_FILES_ONLY=1
export ASR_LANGUAGE=zh
/mnt/F/emotional/miniconda3/envs/himalgg/bin/python app.py
```

若使用新环境，将最后一行改为该环境的 `python app.py`。默认监听 `0.0.0.0:8000`，本机可打开 `http://localhost:8000`。远程浏览器采集媒体通常需要 HTTPS；不应将没有访问控制的开发服务直接暴露到公网。先查看 `/startup_status`，确认各模型错误，而不是只看进度达到 100%；Whisper 是首次转写时懒加载，启动完成不代表 ASR 权重已验证。

需要排查转写时可设置 `ASR_LOG_LEVEL=DEBUG` 后重启，结合 `[BUFFER]`、`[VAD]`、`[ASR]` 日志检查实际输入和片段流转。VAD 配置集中在 `core/asr_config.py`；先确认设备与信号，再根据证据调参。连续下一句调试已按用户要求终止，临时 `.dbg` 配置、日志及网络插桩已清理；真人复测与问题修复仍未确认。既有调试进程的停止状态未能全部核实，不要对外分享原始日志或环境配置。
