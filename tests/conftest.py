"""
GPU도 모델도 없는 환경에서 handler를 import 할 수 있게 무거운 의존성을 스텁으로 채운다.

handler.py는 import 시점에 runpod.serverless.start()를 부르고, batch_transcriber는
faster_whisper.WhisperModel을 import 한다. 둘 다 실제 동작이 필요 없는 테스트라
sys.modules에 가짜를 먼저 꽂아둔다.
"""

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


if "faster_whisper" not in sys.modules:
    _install_stub(
        "faster_whisper",
        WhisperModel=object,
        download_model=lambda *args, **kwargs: None,
    )

if "runpod" not in sys.modules:
    serverless = _install_stub("runpod.serverless", start=lambda *args, **kwargs: None)
    _install_stub("runpod", serverless=serverless)
