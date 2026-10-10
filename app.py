# -*- coding: utf-8 -*-
"""
app.py
功能：Himal 多模态情感识别系统后端主程序
实现：基于 FastAPI 框架，管理 AI 模型生命周期，提供 WebSocket 实时音视频处理接口及文件分析接口
"""
import os
import tempfile

# 在模型依赖导入 Matplotlib 前指定可写缓存，保留用户显式配置。
if not os.environ.get('MPLCONFIGDIR'):
    mpl_cache_dir = os.path.join(tempfile.gettempdir(), f'himalgg-matplotlib-{os.getuid()}')
    os.makedirs(mpl_cache_dir, mode=0o700, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = mpl_cache_dir

# Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import base64
import asyncio
import uuid
from pathlib import Path
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import librosa

from core.vision_pipeline import VisionPipeline
from core.audio_emotion_model import AudioEmotionModel
from core.text_emotion_model import TextEmotionModel, normalize_text
from core.speech_segmenter import SpeechSegmenter
from core.asr_config import (
    SAMPLE_RATE, EMOTION_RING_SECONDS, EMOTION_INTERVAL_SECONDS, log_event,
)
from core.fusion_model import MultimodalFusion

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Trigger loading in background immediately
    print("[System] Triggering startup model loading...")
    asyncio.create_task(asyncio.to_thread(models.load_all))
    yield
    print("[System] Shutting down...")

app = FastAPI(lifespan=lifespan)

# Check for templates directory
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

# Initialize Models Lazily
if torch.cuda.is_available():
    device = 'cuda:0'
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[System] GPU detected: {gpu_name}. Using {device} for acceleration.")
    except:
        print(f"[System] GPU detected. Using {device}.")
else:
    device = 'cpu'
    print("[System] No GPU detected. Using CPU (this may be slow).")

print(f"[System] System will run on {device}")

class ModelContainer:
    def __init__(self):
        self.vision = None
        self.audio = None
        self.text = None
        self.fusion = MultimodalFusion()
        
        self.loading_progress = 0
        self.status_message = "Waiting to start..."
        self.loading_error = None
        self.model_errors = {}
        
        # Internal locks
        self._loading_lock = False

    def load_all(self):
        if self._loading_lock: return
        self._loading_lock = True
        self.loading_error = None
        
        self.loading_progress = 0
        try:
            for name, factory, message, progress in (
                ("vision", VisionPipeline, "Loading Vision Pipeline (YOLOv8 + Xception)...", 30),
                ("audio", AudioEmotionModel, "Loading Audio Model (Wav2Vec2)...", 60),
                ("text", TextEmotionModel, "Loading Text Models (Whisper + RoBERTa)...", 90),
            ):
                self.status_message = message
                print(f"[Startup] {message}")
                try:
                    if getattr(self, name) is None:
                        setattr(self, name, factory(device=device))
                    self.model_errors.pop(name, None)
                except Exception as e:
                    self.model_errors[name] = str(e)
                    print(f"[Startup] {name} Load Error: {e}")
                    import traceback
                    traceback.print_exc()
                self.loading_progress = progress

            self.loading_progress = 100
            if self.model_errors:
                self.loading_error = "; ".join(f"{name}: {error}" for name, error in self.model_errors.items())
                self.status_message = f"Degraded: failed models: {', '.join(self.model_errors)}"
                print(f"[Startup] {self.status_message}")
            else:
                self.status_message = "Ready"
                print("[Startup] All models loaded successfully.")
        finally:
            self._loading_lock = False

    # Deprecated individual loaders (kept for compatibility if needed, but load_all is preferred)
    def load_vision_only(self): pass 
    def load_audio_text(self): pass

models = ModelContainer()



@app.get("/startup_status")
async def startup_status():
    return {
        "status": "loading" if models._loading_lock or models.loading_progress < 100 else ("degraded" if models.model_errors else "ready"),
        "progress": models.loading_progress,
        "message": models.status_message,
        "error": models.loading_error,
        "model_errors": models.model_errors.copy()
    }

@app.post("/init_models")
async def init_models():
    # Trigger global load if not started
    if not models.vision and not models._loading_lock:
        asyncio.create_task(asyncio.to_thread(models.load_all))
    return {"status": "initializing", "message": models.status_message}

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/@vite/client")
async def vite_client_stub():
    return Response("", media_type="application/javascript")

@app.get("/%40vite/client")
async def vite_client_stub_encoded():
    return Response("", media_type="application/javascript")

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def _safe_join_video_path(source: str) -> str:
    p = Path(source)
    if p.is_absolute():
        raise HTTPException(status_code=400, detail="Absolute path not allowed")
    resolved = (PROJECT_ROOT / p).resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return str(resolved)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    upload_id = uuid.uuid4().hex
    save_name = f"{upload_id}{ext}"
    save_path = UPLOADS_DIR / save_name
    
    # Optimize: Use streaming to write file instead of reading all into RAM
    try:
        import shutil
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        file.file.close()
        
    return {"source": f"uploads/{save_name}", "filename": file.filename}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    if not models.vision:
        # Fallback if WS connects before loading done (should be blocked by frontend)
        await websocket.send_json({"status": "loading", "message": "Waiting for models..."})
    
    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()
            
            # Remove auto-trigger since we load on startup now
            if not models.vision:
                await websocket.send_json({"status": "loading", "message": "Models loading..."})
                continue
                
            payload = json.loads(data)
            
            vision_results = []
            if "image" in payload and payload["image"]:
                try:
                    img_data = base64.b64decode(payload["image"].split(',')[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        vision_results = await asyncio.to_thread(models.vision.predict, frame)
                except Exception as e:
                    print(f"[Vision] Processing Error: {e}")

            audio_results = None
            text_results = None
            if "audio" in payload and payload["audio"]:
                # 仅自动加载尚未记录加载失败的缺失模态，避免每个音频包重复重试。
                if any(getattr(models, name) is None and name not in models.model_errors for name in ('audio', 'text')) and not models._loading_lock:
                    asyncio.create_task(asyncio.to_thread(models.load_all))
                try:
                    audio_waveform = np.array(payload["audio"], dtype=np.float32)
                    sample_rate = payload.get("sample_rate", 16000)
                    
                    tasks = []
                    if models.audio:
                        tasks.append(asyncio.to_thread(models.audio.predict, audio_waveform, sample_rate))
                    else:
                        tasks.append(asyncio.sleep(0))
                        
                    if models.text:
                        tasks.append(asyncio.to_thread(models.text.predict, audio_waveform, sample_rate))
                    else:
                        tasks.append(asyncio.sleep(0))
                        
                    results = await asyncio.gather(*tasks)
                    
                    # Only update if we got valid results
                    if results[0]: audio_results = results[0]
                    if results[1]: text_results = results[1]
                    
                except Exception as e:
                    print(f"[Audio/Text] Processing Error: {e}")

            v_probs = vision_results[0]["probs"] if vision_results else None
            a_probs = audio_results["probs"] if audio_results else None
            t_probs = text_results["probs"] if text_results else None
            
            fusion_result = models.fusion.fuse(
                vision_probs=v_probs,
                audio_probs=a_probs,
                text_probs=t_probs
            )
            
            response = {
                "vision": vision_results,
                "audio": audio_results,
                "text": text_results,
                "fusion": fusion_result,
                "timestamp": time.time(),
                "total_latency": time.time() - start_time
            }
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.websocket("/ws_realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()
    audio_buf_16k = np.zeros((0,), dtype=np.float32)  # Independent emotion-only ring.
    segmenter = SpeechSegmenter()
    asr_queue = asyncio.Queue()  # Strategy A: unbounded, never evict pending segments.
    queued_samples = 0
    last_text_value = None
    last_text_payload = None
    last_audio_payload = None
    last_audio_infer_ts = 0.0
    is_transcribing = False
    vad_error = None
    asr_worker_error = None
    text_error = None
    asr_completed_segments = 0
    asr_last_result = None
    received_audio_chunks = 0
    received_audio_seconds = 0.0
    buffer_log_ts = time.monotonic()
    buffer_samples = 0
    buffer_seconds = 0.0
    buffer_sum_squares = 0.0
    buffer_peak = 0.0
    buffer_sample_rates = set()

    def record_received_audio(waveform, sample_rate):
        nonlocal received_audio_chunks, received_audio_seconds
        nonlocal buffer_samples, buffer_seconds, buffer_sum_squares, buffer_peak
        received_audio_chunks += 1
        duration = waveform.size / sample_rate
        received_audio_seconds += duration
        buffer_samples += waveform.size
        buffer_seconds += duration
        buffer_sum_squares += float(np.sum(np.square(waveform, dtype=np.float64)))
        buffer_peak = max(buffer_peak, float(np.max(np.abs(waveform))))
        buffer_sample_rates.add(sample_rate)

    def enqueue_segments(segments):
        nonlocal queued_samples
        for segment in segments:
            asr_queue.put_nowait(segment)
            queued_samples += segment.audio.size
            log_event('VAD][BUFFER', f'enqueue start={segment.start_sample / SAMPLE_RATE:.3f}s '
                      f'end={segment.end_sample / SAMPLE_RATE:.3f}s duration={segment.audio.size / SAMPLE_RATE:.3f}s '
                      f'reason={segment.reason} pending={asr_queue.qsize()} pending_seconds={queued_samples / SAMPLE_RATE:.3f}')

    async def run_asr_worker():
        nonlocal is_transcribing, last_text_payload, last_text_value, queued_samples, asr_worker_error, text_error
        nonlocal asr_completed_segments, asr_last_result
        while True:
            segment = await asr_queue.get()
            try:
                if segment is None:
                    return
                is_transcribing = True
                # No asr_pipeline guard: transcription itself triggers lazy loading.
                while models.text is None and models._loading_lock:
                    await asyncio.sleep(0.1)
                if models.text is None:
                    raise RuntimeError('Text model unavailable')
                log_event('ASR', f'dequeue start={segment.start_sample / SAMPLE_RATE:.3f}s '
                          f'end={segment.end_sample / SAMPLE_RATE:.3f}s reason={segment.reason}')
                asr_worker_error = None
                res = await asyncio.to_thread(models.text.transcribe_segment, segment.audio)
                asr_completed_segments += 1
                asr_last_result = 'text' if res else 'empty'
                if res:
                    last_text_value = res
                    # 转写先独立保存，情绪推理失败也不能丢失已经识别的文本。
                    last_text_payload = {"text": res, "emotion": None, "probs": {}}
                    t_res = await asyncio.to_thread(models.text.predict_from_text, res)
                    text_error = models.text.last_predict_error
                    if t_res:
                        last_text_payload = t_res
            except Exception as e:
                asr_worker_error = str(e)
                log_event('ASR', f'worker_error={e}')
            finally:
                if segment is not None:
                    queued_samples -= segment.audio.size
                is_transcribing = False
                asr_queue.task_done()

    worker = asyncio.create_task(run_asr_worker())
    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()
            payload = json.loads(data)
            vision_results = []
            vision_error = models.model_errors.get("vision") if not models.vision else None
            if payload.get("image") and not models.vision:
                vision_error = vision_error or 'Vision model not ready'
            elif payload.get("image"):
                try:
                    img_data = base64.b64decode(payload["image"].split(",")[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        vision_results = await asyncio.to_thread(models.vision.predict, frame)
                except Exception as e:
                    vision_error = str(e)
                    print(f"[Realtime][Vision] Error: {e}")
            audio_results = None
            text_results = None
            audio_error = models.audio.last_predict_error if models.audio else None
            audio_in_samples = 0
            audio_in_bytes = 0
            audio_in_sr = payload.get("sample_rate", 16000)

            if "audio_b64" in payload and payload["audio_b64"]:
                # 仅自动加载尚未记录加载失败的缺失模态，已知失败不随音频包重试。
                if any(getattr(models, name) is None and name not in models.model_errors for name in ('audio', 'text')) and not models._loading_lock:
                    print("[WS] Models missing without known load errors, triggering load...")
                    asyncio.create_task(asyncio.to_thread(models.load_all))
                
                try:
                    raw = base64.b64decode(payload["audio_b64"])
                    audio_in_bytes = len(raw)
                    audio_i16 = np.frombuffer(raw, dtype=np.int16)
                    audio_in_samples = int(audio_i16.size)
                    if audio_i16.size > 0:
                        audio_waveform = (audio_i16.astype(np.float32) / 32768.0)
                        sample_rate = payload.get("sample_rate", 16000)
                        record_received_audio(audio_waveform, sample_rate)
                        if sample_rate != 16000:
                            audio_waveform = librosa.resample(audio_waveform, orig_sr=sample_rate, target_sr=16000).astype(np.float32)
                            sample_rate = 16000

                        # 先写独立情绪缓冲，VAD失败只影响转写链路。
                        audio_buf_16k = np.concatenate([audio_buf_16k, audio_waveform])[-int(SAMPLE_RATE * EMOTION_RING_SECONDS):]
                        try:
                            enqueue_segments(await asyncio.to_thread(segmenter.feed, audio_waveform))
                            vad_error = None
                        except Exception as e:
                            vad_error = str(e)
                            log_event('VAD', f'feed_failed={e}')
                except Exception as e:
                    audio_error = str(e)
                    print(f"[Audio/Text] Error: {e}")

            elif "audio" in payload and payload["audio"]:
                # Prefer audio_b64 when both encodings are present; feed each sample once.
                if any(getattr(models, name) is None and name not in models.model_errors for name in ('audio', 'text')) and not models._loading_lock:
                    asyncio.create_task(asyncio.to_thread(models.load_all))
                try:
                    audio_waveform = np.array(payload["audio"], dtype=np.float32)
                    sample_rate = payload.get("sample_rate", 16000)
                    audio_in_samples = int(audio_waveform.size)
                    record_received_audio(audio_waveform, sample_rate)
                    if sample_rate != 16000:
                        audio_waveform = librosa.resample(audio_waveform, orig_sr=sample_rate, target_sr=16000).astype(np.float32)
                        sample_rate = 16000

                    audio_buf_16k = np.concatenate([audio_buf_16k, audio_waveform])[-int(SAMPLE_RATE * EMOTION_RING_SECONDS):]
                    try:
                        enqueue_segments(await asyncio.to_thread(segmenter.feed, audio_waveform))
                        vad_error = None
                    except Exception as e:
                        vad_error = str(e)
                        log_event('VAD', f'feed_failed={e}')
                except Exception as e:
                    audio_error = str(e)
                    print(f"[Audio/Text] Error: {e}")

            buffer_now = time.monotonic()
            if buffer_now - buffer_log_ts >= 5.0:
                rms = (buffer_sum_squares / buffer_samples) ** 0.5 if buffer_samples else 0.0
                log_event('BUFFER', f'received_chunks={received_audio_chunks} received_total_sec={received_audio_seconds:.3f} '
                          f'window_wall_sec={buffer_now - buffer_log_ts:.3f} window_audio_sec={buffer_seconds:.3f} '
                          f'samples={buffer_samples} sample_rates={sorted(buffer_sample_rates)} rms={rms:.6f} peak={buffer_peak:.6f} '
                          f'vad_buffer_sec={segmenter.buffered_samples / SAMPLE_RATE:.3f} '
                          f'asr_pending={asr_queue.qsize()} asr_pending_sec={queued_samples / SAMPLE_RATE:.3f}')
                buffer_log_ts = buffer_now
                buffer_samples = 0
                buffer_seconds = 0.0
                buffer_sum_squares = 0.0
                buffer_peak = 0.0
                buffer_sample_rates.clear()

            if models.audio and audio_buf_16k.size >= int(SAMPLE_RATE * EMOTION_RING_SECONDS):
                try:
                    now = time.time()
                    if (now - last_audio_infer_ts) >= EMOTION_INTERVAL_SECONDS:
                        chunk = audio_buf_16k.copy()
                        audio_results = await asyncio.to_thread(models.audio.predict, chunk, SAMPLE_RATE)
                        audio_error = models.audio.last_predict_error
                        if audio_results:
                            last_audio_payload = audio_results
                        last_audio_infer_ts = now
                    else:
                        audio_results = last_audio_payload
                except Exception as e:
                    audio_error = str(e)
                    print(f"[Audio/Text] Error: {e}")

            # ASR / Text Logic
            # 1. Check if client sent text directly (Browser ASR)
            if "text_input" in payload and payload["text_input"]:
                last_text_value = normalize_text(payload["text_input"])
                # Run emotion analysis on this text immediately
                if models.text:
                    try:
                        res = await asyncio.to_thread(models.text.predict, last_text_value)
                        text_error = models.text.last_predict_error
                        if res:
                            last_text_payload = res
                        else:
                            last_text_payload = {"text": last_text_value, "emotion": "neutral", "probs": {}}
                    except Exception as e:
                        print(f"[Text] Analysis Error: {e}")
                        last_text_payload = {"text": last_text_value, "emotion": "neutral", "probs": {}}
                else:
                     last_text_payload = {"text": last_text_value, "emotion": "neutral", "probs": {}}
                
                # Browser text does not reset or evict the independent ASR queue.

            if text_results is None and last_text_payload is not None:
                text_results = last_text_payload
            
            v_probs = vision_results[0]["probs"] if vision_results else None
            a_probs = audio_results["probs"] if audio_results else None
            if text_results is None and last_text_payload is not None:
                text_results = last_text_payload
            t_probs = text_results["probs"] if text_results and isinstance(text_results, dict) else None
            
            # Use previous results if current frame has none, to avoid flickering
            # Ideally we'd have a stateful tracker, but for now simple persistence:
            # (Note: This is stateless per request, so persistent state needs to be outside loop or client-side)
            # Actually, client updates text only when non-null. Backend sends null if no speech.
            
            fusion_result = models.fusion.fuse(
                vision_probs=v_probs,
                audio_probs=a_probs,
                text_probs=t_probs
            )
            
            response = {
                "vision": vision_results,
                "audio": audio_results,
                "text": text_results,
                "fusion": fusion_result,
                "debug": {
                    "vision_faces": len(vision_results) if vision_results else 0,
                    "audio_buf_sec": float(audio_buf_16k.size) / 16000.0,
                    "asr_completed_segments": asr_completed_segments,
                    "asr_last_result": asr_last_result,
                    "asr_pending_segments": asr_queue.qsize(),
                    "asr_pending_sec": queued_samples / SAMPLE_RATE,
                    "vad_buffer_sec": segmenter.buffered_samples / SAMPLE_RATE,
                    "has_text_model": models.text is not None,
                    "asr_loaded": models.text.asr_pipeline is not None if models.text else False,
                    "asr_loading": models.text._loading_asr if models.text else False,
                    "asr_error": asr_worker_error or (models.text.asr_load_error or models.text.asr_infer_error if models.text else None),
                    "asr_load_error": models.text.asr_load_error if models.text else None,
                    "asr_infer_error": models.text.asr_infer_error if models.text else None,
                    "vad_error": vad_error,
                    "models_loading": models._loading_lock,
                    "model_load_error": models.loading_error,
                    "text_load_error": models.text.emotion_load_error if models.text else None,
                    "has_audio_model": models.audio is not None,
                    "last_text": last_text_value,
                    "has_vision_model": models.vision is not None,
                    "vision_error": vision_error,
                    "audio_error": audio_error,
                    "text_error": text_error,
                    "audio_in_samples": audio_in_samples,
                    "audio_in_bytes": audio_in_bytes,
                    "audio_in_sr": audio_in_sr,
                    "received_audio_chunks": received_audio_chunks,
                    "received_audio_sec": received_audio_seconds,
                    "is_transcribing": is_transcribing,
                },
                "timestamp": time.time(),
                "total_latency": time.time() - start_time
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            enqueue_segments(await asyncio.to_thread(segmenter.flush))
        except Exception as e:
            log_event('VAD', f'disconnect_flush_failed={e}')
        # Drain strategy A before stopping: cancelling to_thread cannot stop inference.
        asr_queue.put_nowait(None)
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            await worker
            raise
        finally:
            log_event('BUFFER', f'disconnect drained pending={asr_queue.qsize()}')
            try:
                await websocket.close()
            except Exception:
                pass

@app.websocket("/ws_server_video")
async def ws_server_video(websocket: WebSocket):
    await websocket.accept()
    cap = None
    text_worker = None
    stopping = False
    playback_epoch = 0
    text_queue = asyncio.Queue()
    audio_results = None
    text_results = None
    text_error = None
    is_transcribing = False

    async def run_text_worker():
        nonlocal text_results, text_error, is_transcribing
        while True:
            item = await text_queue.get()
            try:
                if item is None:
                    return
                epoch, segment = item
                if stopping or epoch != playback_epoch:
                    continue
                is_transcribing = True
                while models.text is None and models._loading_lock and not stopping and epoch == playback_epoch:
                    await asyncio.sleep(0.1)
                if stopping or epoch != playback_epoch:
                    continue
                if models.text is None:
                    raise RuntimeError('Text model not ready')
                transcript = await asyncio.to_thread(models.text.transcribe_segment, segment.audio, SAMPLE_RATE)
                if stopping or epoch != playback_epoch:
                    continue
                if transcript:
                    text_results = {"text": transcript, "emotion": None, "probs": {}}
                    result = await asyncio.to_thread(models.text.predict_from_text, transcript)
                    if not stopping and epoch == playback_epoch:
                        if result:
                            text_results = result
                        text_error = models.text.last_predict_error
                elif not stopping and epoch == playback_epoch:
                    text_error = models.text.asr_load_error or models.text.asr_infer_error
            except Exception as e:
                if not stopping and epoch == playback_epoch:
                    text_error = str(e)
            finally:
                is_transcribing = False
                text_queue.task_done()

    def decode_video_audio(path):
        import av
        from faster_whisper.audio import decode_audio
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError('Video has no audio track')
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE, split_stereo=False)
        if not audio.size:
            raise ValueError('Video audio track decoded to zero samples')
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def prepare_speech_segments(audio):
        segmenter = SpeechSegmenter()
        return segmenter.feed(audio) + segmenter.flush()

    try:
        init_source = None
        init_multimodal = True
        try:
            init_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            try:
                init_payload = json.loads(init_msg)
                init_source = init_payload.get("source")
                init_multimodal = bool(init_payload.get("multimodal", True))
            except Exception:
                init_source = None
        except Exception:
            init_source = None

        if not models.vision and not models._loading_lock:
            asyncio.create_task(asyncio.to_thread(models.load_all))
        
        while not models.vision:
            if models.loading_error and not models._loading_lock:
                await websocket.send_json({"error": models.loading_error, "debug": {"vision_error": models.loading_error}})
                return
            await websocket.send_json({"status": "loading", "message": models.status_message})
            await asyncio.sleep(0.2)

        source = init_source or os.getenv("VIDEO_SOURCE", "static/test_demo.mp4")
        cap_source = int(source) if str(source).isdigit() else _safe_join_video_path(str(source))
        print(f"[WS Server] Opening video source: {cap_source}")
        
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            err_msg = f"Cannot open VIDEO_SOURCE={source}"
            print(f"[WS Server] Error: {err_msg}")
            await websocket.send_json({"error": err_msg})
            # Give frontend time to receive message before closing
            await asyncio.sleep(1.0)
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0

        audio_ready = False
        audio_error = None
        audio_y = None
        audio_sr = SAMPLE_RATE
        speech_segments = []
        text_cursor = 0
        if init_multimodal and isinstance(cap_source, str):
            if any(getattr(models, name) is None and name not in models.model_errors for name in ('audio', 'text')) and not models._loading_lock:
                asyncio.create_task(asyncio.to_thread(models.load_all))
            try:
                # PyAV直接解码MP4音轨为16k单声道，不依赖librosa的容器支持。
                audio_y = await asyncio.to_thread(decode_video_audio, cap_source)
                audio_ready = True
                print(f"[WS Server] Audio loaded successfully ({len(audio_y)/audio_sr:.1f}s)")
            except Exception as e:
                audio_error = f"Audio load failed: {str(e)}"
                print(f"[WS Server] Audio Warning: {audio_error}")
                # Don't crash, just proceed without audio


        if audio_ready:
            try:
                # 这里只做整句VAD，转写必须等播放到片段末尾后由后台串行执行。
                speech_segments = await asyncio.to_thread(prepare_speech_segments, audio_y)
            except Exception as e:
                text_error = f'VAD segmentation failed: {e}'
            text_worker = asyncio.create_task(run_text_worker())

        last_audio_ts = -1.0

        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not cap.isOpened() or frame_idx == 0:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    last_audio_ts = -1.0
                    text_cursor = 0
                    playback_epoch += 1
                    audio_results = None
                    text_results = None
                    while not text_queue.empty():
                        text_queue.get_nowait()
                        text_queue.task_done()
                    # 不取消正在运行的线程；旧轮次结果会被丢弃，同一个worker继续串行执行。
                    await asyncio.sleep(0.03)
                    continue

                frame_h, frame_w = frame.shape[:2]
                start_time = time.time()
                vision_results = []
                vision_error = None
                try:
                    vision_results = await asyncio.to_thread(models.vision.predict, frame)
                except Exception as e:
                    vision_error = str(e)

                v_probs = vision_results[0]["probs"] if vision_results else None
                t_sec = frame_idx / fps
                if audio_ready and audio_y is not None:
                    if (t_sec - last_audio_ts) >= 1.0:
                        last_audio_ts = t_sec
                        if models.audio is None:
                            audio_error = models.model_errors.get("audio") or 'Audio model not ready'
                        elif models.audio.classifier is None:
                            audio_error = 'Audio classifier unavailable; heuristic fallback disabled for Demo'
                        else:
                            s0 = int(max(0, t_sec * audio_sr))
                            s1 = int(min(len(audio_y), s0 + audio_sr))
                            chunk = audio_y[s0:s1].astype(np.float32)
                            if chunk.size >= int(0.5 * audio_sr):
                                try:
                                    result = await asyncio.to_thread(models.audio.predict, chunk, audio_sr)
                                    if result is None:
                                        raise RuntimeError('Audio inference returned no result')
                                    audio_results = result
                                    audio_error = None
                                except Exception as e:
                                    audio_error = str(e)
                    if models.text is None and text_error is None:
                        text_error = models.model_errors.get("text") or 'Text model not ready'
                    while text_cursor < len(speech_segments):
                        segment = speech_segments[text_cursor]
                        # 最后一帧覆盖到下一帧时间，避免末尾flush片段永远无法调度。
                        if segment.end_sample / audio_sr > (frame_idx + 1) / fps:
                            break
                        text_queue.put_nowait((playback_epoch, segment))
                        text_cursor += 1

                a_probs = audio_results["probs"] if audio_results else None
                t_probs = text_results["probs"] if text_results else None
                fusion_result = models.fusion.fuse(vision_probs=v_probs, audio_probs=a_probs, text_probs=t_probs) if any((v_probs, a_probs, t_probs)) else None

                ok_jpg, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ok_jpg:
                    frame_idx += 1
                    continue
                b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")

                await websocket.send_json({
                    "mode": "server_video",
                    "frame": f"data:image/jpeg;base64,{b64}",
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                    "vision": vision_results,
                    "audio": audio_results,
                    "text": text_results,
                    "fusion": fusion_result,
                    "audio_error": audio_error,
                    "debug": {
                        "has_vision_model": models.vision is not None,
                        "vision_error": vision_error,
                        "audio_error": audio_error,
                        "text_error": text_error,
                        "has_audio_model": models.audio is not None,
                        "has_text_model": models.text is not None,
                        "audio_ready": audio_ready,
                        "models_loading": models._loading_lock,
                        "model_load_error": models.loading_error,
                        "asr_loading": models.text._loading_asr if models.text else False,
                        "asr_error": models.text.asr_load_error or models.text.asr_infer_error if models.text else None,
                        "text_load_error": models.text.emotion_load_error if models.text else models.model_errors.get("text"),
                        "asr_pending_segments": text_queue.qsize(),
                        "is_transcribing": is_transcribing,
                    },
                    "timestamp": time.time(),
                    "total_latency": time.time() - start_time
                })
                frame_idx += 1
                
                # Dynamic sleep to match target FPS
                process_time = time.time() - start_time
                target_frame_time = 1.0 / fps
                sleep_time = max(0.001, target_frame_time - process_time)
                await asyncio.sleep(sleep_time)
        finally:
            cap.release()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        if cap is not None:
            cap.release()
        stopping = True
        playback_epoch += 1
        if text_worker is not None:
            while not text_queue.empty():
                text_queue.get_nowait()
                text_queue.task_done()
            text_queue.put_nowait(None)
            # to_thread取消无法停止推理，等待当前工作结束，防止遗留线程重复调用共享模型。
            try:
                await asyncio.shield(text_worker)
            except asyncio.CancelledError:
                await text_worker
                raise
        try:
            await websocket.close()
        except Exception:
            pass

@app.websocket("/ws_video")
async def ws_video(websocket: WebSocket):
    await ws_server_video(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
