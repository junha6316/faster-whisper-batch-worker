"""handler 전체 흐름 테스트. GPU 대신 가짜 transcriber를 꽂는다."""

import asyncio
import base64
import os

import pytest

import handler


def _run(job_input, job_id="job-1"):
    return asyncio.run(handler.handler({"id": job_id, "input": job_input}))


def _b64(payload: bytes = b"audio") -> str:
    return base64.b64encode(payload).decode()


class _FakeTranscriber:
    def __init__(self):
        self.calls = []

    def transcribe_batch(self, audio_paths, **kwargs):
        self.calls.append({"paths": list(audio_paths), **kwargs})
        return [
            {
                "transcription": f"text-{index}",
                "segments": [{"start": 0.0, "end": 1.0, "text": f"text-{index}"}],
                "inference_time": 0.1,
            }
            for index, _ in enumerate(audio_paths)
        ]


@pytest.fixture
def fake_transcriber(monkeypatch):
    fake = _FakeTranscriber()
    monkeypatch.setattr(handler, "get_transcriber", lambda: fake)
    return fake


class TestInputErrors:
    def test_오디오가_없으면_에러(self):
        assert "No audio data provided" in _run({})["error"]

    def test_지원하지_않는_output_format은_에러(self):
        result = _run({"audio_base64": _b64(), "output_formats": ["docx"]})
        assert "Unsupported output_formats" in result["error"]

    def test_타입이_틀리면_traceback_대신_에러_메시지(self):
        # 핸들러 밖으로 예외가 나가면 runpod SDK가 traceback과 호스트 정보를 응답에 싣는다.
        result = _run({"audio_urls": 42})
        assert result["error"].startswith("Invalid input parameter")

    def test_beam_size가_숫자가_아니면_에러(self):
        result = _run({"audio_base64": _b64(), "beam_size": "fast"})
        assert result["error"].startswith("Invalid input parameter")


class TestAllInputsFailed:
    def test_전부_실패하면_GPU를_건드리지_않는다(self, monkeypatch):
        monkeypatch.setattr(
            handler, "_materialize_url", lambda url: {"error": "download failed: 404"}
        )

        def _explode():
            raise AssertionError("전부 실패한 job에서 모델을 올리면 안 된다")

        monkeypatch.setattr(handler, "get_transcriber", _explode)

        result = _run({"audio_urls": ["https://e.com/a.wav", "https://e.com/b.wav"]})
        assert len(result["results"]) == 2
        assert all("download failed" in r["error"] for r in result["results"])


class TestSuccessPath:
    def test_배치_결과가_입력_순서대로_돌아온다(self, fake_transcriber):
        result = _run({"audio_base64_list": [_b64(b"a"), _b64(b"b")]})
        assert [r["transcription"] for r in result["results"]] == ["text-0", "text-1"]

    def test_임시파일이_정리된다(self, fake_transcriber):
        _run({"audio_base64_list": [_b64(b"a"), _b64(b"b")]})
        paths = fake_transcriber.calls[0]["paths"]
        assert paths and not any(os.path.exists(path) for path in paths)

    def test_srt가_붙는다(self, fake_transcriber):
        result = _run({"audio_base64": _b64(), "output_formats": ["text", "srt"]})
        assert result["results"][0]["srt"].startswith("1\n00:00:00,000 --> 00:00:01,000")

    def test_beam_size_상한이_transcriber까지_전달된다(self, fake_transcriber):
        _run({"audio_base64": _b64(), "beam_size": 500})
        assert fake_transcriber.calls[0]["beam_size"] == handler.MAX_BEAM_SIZE

    def test_language_미지정이면_자동감지로_넘어간다(self, fake_transcriber):
        _run({"audio_base64": _b64()})
        assert fake_transcriber.calls[0]["language"] is None

    def test_일부_실패해도_나머지_배치는_전사된다(self, fake_transcriber, monkeypatch):
        monkeypatch.setattr(
            handler, "_materialize_url", lambda url: {"error": "download failed: 404"}
        )
        result = _run(
            {
                "audio_base64_list": [_b64(b"a")],
                "audio_urls": ["https://e.com/dead.wav"],
                "output_formats": ["srt"],
            }
        )
        assert result["results"][0]["transcription"] == "text-0"
        assert "download failed" in result["results"][1]["error"]
        # 실패 슬롯에는 자막 키가 붙지 않는다.
        assert "srt" not in result["results"][1]
        # 성공한 파일만 GPU로 갔다.
        assert len(fake_transcriber.calls[0]["paths"]) == 1
