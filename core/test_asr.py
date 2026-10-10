"""Offline regression checks: no model constructors and no weight downloads."""
import ast
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from core.asr_config import SAMPLE_RATE, VAD_WINDOW, is_valid_transcript
from core.speech_segmenter import SpeechSegmenter


class SegmenterTests(unittest.TestCase):
    def segmenter(self):
        return SpeechSegmenter(probability=lambda window: float(window[0] > 0))

    def test_arbitrary_packets_and_normal_nonoverlap(self):
        s = self.segmenter()
        audio = np.concatenate((np.ones(16 * VAD_WINDOW), np.zeros(20 * VAD_WINDOW),
                                np.ones(16 * VAD_WINDOW), np.zeros(20 * VAD_WINDOW))).astype(np.float32)
        segments = []
        for offset in range(0, audio.size, 137):
            segments += s.feed(audio[offset:offset + 137])
        segments += s.flush()
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(item.reason == 'silence' for item in segments))
        self.assertLessEqual(segments[0].end_sample, segments[1].start_sample)

    def test_minimum_and_exact_tail(self):
        s = self.segmenter()
        self.assertEqual(s.feed(np.ones(3999, dtype=np.float32)) + s.flush(), [])
        s = self.segmenter()
        segments = s.feed(np.ones(4000, dtype=np.float32)) + s.flush()
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].end_sample, 4000)

    def test_silence_timeout_and_padding(self):
        s = self.segmenter()
        self.assertEqual(s.feed(np.ones(8 * VAD_WINDOW, dtype=np.float32)), [])
        self.assertEqual(s.feed(np.zeros(15 * VAD_WINDOW, dtype=np.float32)), [])
        segments = s.feed(np.zeros(VAD_WINDOW, dtype=np.float32))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].end_sample, 8 * VAD_WINDOW + 1600)

    def test_forced_overlap_only(self):
        s = self.segmenter()
        segments = s.feed(np.ones(31 * SAMPLE_RATE, dtype=np.float32)) + s.flush()
        self.assertGreaterEqual(len(segments), 3)
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(previous.reason, 'max_duration')
            self.assertEqual(previous.end_sample - current.start_sample, 4000)
            self.assertLessEqual(previous.audio.size, 15 * SAMPLE_RATE + VAD_WINDOW + 1600)

    def test_no_duplicate_overlap_tail(self):
        s = self.segmenter()
        segments = s.feed(np.ones(469 * VAD_WINDOW, dtype=np.float32)) + s.flush()
        self.assertEqual(len(segments), 1)

    def test_connection_isolation_and_bounded_silence(self):
        first, second = self.segmenter(), self.segmenter()
        first.feed(np.ones(5000, dtype=np.float32))
        second.feed(np.zeros(50 * SAMPLE_RATE, dtype=np.float32))
        self.assertEqual(second.flush(), [])
        self.assertEqual(len(first.flush()), 1)
        self.assertLessEqual(second.buffered_samples, 1600 + VAD_WINDOW)

    def test_filter(self):
        for value in ('', ' ', '!!!', '[BLANK_AUDIO]', '[ Silence ]', '(music)', '♪', 'a'):
            self.assertFalse(is_valid_transcript(value), value)
        for value in ('你', '你好', '好的', 'thank you', '你好好的', '好好好好好好', 'OK'):
            self.assertTrue(is_valid_transcript(value), value)


class MicrophoneTests(unittest.TestCase):
    def test_device_selection_and_stale_permission(self):
        import shutil
        import subprocess
        node = shutil.which('node')
        if not node:
            self.skipTest('Node.js required for browser media regression')
        html = (Path(__file__).resolve().parents[1] / 'templates/index.html').read_text(encoding='utf-8')
        script = html.split('<script>')[1].split('</script>')[0]
        syntax = subprocess.run([node, '--check'], input=script, text=True, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        selection = script.split("        const microphoneSelect =")[1].split('        let lastFaces')[0]
        selection = 'const microphoneSelect =' + selection
        media = script.split('        function isCurrentMediaSession')[1].split('        function handleStream')[0]
        media = 'function isCurrentMediaSession' + media
        harness = r'''
const assert = require('assert');
const select = {value:'', options:[], disabled:true,
    replaceChildren(...items) {this.options=items;}, add(item) {this.options.push(item);},
    addEventListener(event, callback) {this.change=callback;}};
const error = {innerText:''};
const document = {getElementById(id) {return id === 'microphone-select' ? select : error;}};
function Option(label, value) {this.label=label;this.value=value;}
const WebSocket = {OPEN:1};
let mediaSessionGeneration=1, wsRealtime={readyState:1}, isConnected=true, globalKillSwitch=false;
let serverMode=false, audioOnlyMode=false, audioContext=null, recognition=null;
let restarts=0;
function startRealtime() {restarts++;}
let streams=[];
function handleStream(stream) {streams.push(stream);}
const window = {AudioContext:class {constructor(){this.state='running';} }};
const navigator = {mediaDevices:{}};
'''
        assertions = r'''
(async () => {
    assert(!microphoneConstraints().deviceId);
    navigator.mediaDevices.enumerateDevices = async () => [
        {kind:'audioinput',label:'chosen mic',deviceId:'chosen'},
        {kind:'videoinput',label:'camera',deviceId:'camera'}];
    await refreshMicrophones(1,wsRealtime);
    assert.equal(select.options.length,2);
    assert.equal(select.disabled,false);
    select.value='chosen';select.change();assert.equal(restarts,1);
    assert.deepEqual(microphoneConstraints().deviceId,{exact:'chosen'});
    let requests=[];
    navigator.mediaDevices.getUserMedia = async c => {
        requests.push(c); if(c.video) throw new Error('no camera');
        return {getTracks:()=>[]};
    };
    await setupMedia(1,wsRealtime);
    assert.equal(streams.length,1);
    assert.equal(audioOnlyMode,true);
    assert(requests.every(c=>c.audio.deviceId.exact==='chosen'));
    navigator.mediaDevices.getUserMedia=async () => {throw new Error('device unavailable');};
    await setupMedia(1,wsRealtime);
    assert(error.innerText.includes('不会自动替换指定设备'));
    assert.equal(select.disabled,false);
    assert.equal(streams.length,1);
    let stopped=0;
    navigator.mediaDevices.getUserMedia=async () => {
        mediaSessionGeneration++;
        return {getTracks:()=>[{stop(){stopped++;}}]};
    };
    await setupMedia(1,wsRealtime);
    assert.equal(stopped,1);assert.equal(streams.length,1);
    const count=select.options.length;
    await refreshMicrophones(1,wsRealtime);
    assert.equal(select.options.length,count);
    globalKillSwitch=true;select.change();assert.equal(restarts,1);
    console.log('microphone exact/fallback/stale/stop regression passed');
})().catch(err=>{console.error(err);process.exitCode=1;});
'''
        state_logic = script.split("                let state = debug.received_audio_chunks")[1].split('                asrInfo.innerText')[0]
        state_logic = 'let state = debug.received_audio_chunks' + state_logic
        status_checks = '''
const cases = [
    [{has_text_model:true,received_audio_chunks:1}, 'VAD 尚未检出有效语音段'],
    [{has_text_model:true,asr_last_result:'empty'}, 'ASR 已处理语音段，但识别为空'],
    [{has_text_model:true,asr_load_error:'failed'}, 'ASR 加载失败'],
    [{has_text_model:true,is_transcribing:true}, '识别中']];
for (const [debug, expected] of cases) {
    const asrError = debug.asr_load_error;
''' + state_logic + '\nassert.equal(state,expected);}\n'
        result = subprocess.run([node], input=harness + selection + media + status_checks + assertions,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class CompatibilityTests(unittest.TestCase):
    def test_lazy_predict_and_serialized_generator(self):
        # Extract the actual class without importing transformers or emotion weights.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'core/text_emotion_model.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        import threading
        import time
        from core import asr_config
        from opencc import OpenCC
        converter = OpenCC('t2s')
        namespace = dict(vars(asr_config), threading=threading, time=time, np=np,
                         normalize_text=lambda text: converter.convert(text.strip()))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'text_emotion_model.py', 'exec'), namespace)
        model_class = namespace['TextEmotionModel']
        model = model_class.__new__(model_class)
        model.asr_pipeline = None
        model._predict_text_emotion = lambda text: {'text': text}
        model.transcribe = lambda audio, sample_rate=16000: '你好'
        self.assertEqual(model.predict(np.ones(4000))['text'], '你好')
        class FakeModel:
            def transcribe(self, audio, **kwargs):
                def generate():
                    self.assert_locked()
                    from types import SimpleNamespace
                    yield SimpleNamespace(text='這個世界讓我很開心', start=0, end=0.25, avg_logprob=-0.1, no_speech_prob=0.01)
                return generate(), None
            def assert_locked(self):
                if not model_class._asr_lock._is_owned():
                    raise AssertionError('lazy generator escaped inference lock')
        model.load_local_asr = lambda: FakeModel()
        self.assertEqual(model.transcribe_segment(np.ones(4000)), '这个世界让我很开心')

    def test_queue_strategy_a_and_disconnect(self):
        self.exercise_queue(False)

    def test_empty_asr_status(self):
        self.exercise_queue(True)

    def exercise_queue(self, empty):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'app.py').read_text(encoding='utf-8'))
        endpoint = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'websocket_realtime')
        worker = next(node for node in endpoint.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run_asr_worker')
        enqueue = next(node for node in endpoint.body if isinstance(node, ast.FunctionDef) and node.name == 'enqueue_segments')
        outer = ast.AsyncFunctionDef(name='exercise', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.parse('asr_queue = asyncio.Queue()\nqueued_samples = 0\nis_transcribing = False\nlast_text_payload = None\nlast_text_value = None\nasr_worker_error = None\ntext_error = None\nasr_completed_segments = 0\nasr_last_result = None').body +
            [enqueue, worker] + ast.parse('task = asyncio.create_task(run_asr_worker())\nenqueue_segments(segments)\nasr_queue.put_nowait(None)\nawait task\nassert queued_samples == 0\nassert asr_queue.empty()\nassert not is_transcribing\nassert last_text_value == (None if empty else "2")\nassert asr_completed_segments == 3\nassert asr_last_result == ("empty" if empty else "text")').body,
            decorator_list=[])
        from types import SimpleNamespace
        from core.speech_segmenter import SpeechSegment
        from core.asr_config import log_event
        seen = []
        def transcribe(audio):
            seen.append(int(audio[0]))
            return None if empty else str(int(audio[0]))
        model = SimpleNamespace(transcribe_segment=transcribe, predict_from_text=lambda text: {'text': text}, last_predict_error=None)
        namespace = dict(asyncio=asyncio, SAMPLE_RATE=SAMPLE_RATE, log_event=log_event, empty=empty,
                         models=SimpleNamespace(text=model, _loading_lock=False),
                         segments=[SpeechSegment(np.full(4000, i), i * 4000, (i + 1) * 4000, 'flush') for i in range(3)])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[])), 'queue_test', 'exec'), namespace)
        asyncio.run(namespace['exercise']())
        self.assertEqual(seen, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
