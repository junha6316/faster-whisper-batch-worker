"""job input 정규화 테스트. beam_size 상한은 그대로 GPU 과금이라 특히 촘촘히 본다."""

import pytest

import handler


class TestAsList:
    def test_복수형_리스트를_그대로_쓴다(self):
        assert handler._as_list(["a", "b"], None, "audio_url") == ["a", "b"]

    def test_복수형이_비면_단수형을_쓴다(self):
        assert handler._as_list([], "a", "audio_url") == ["a"]
        assert handler._as_list(None, "a", "audio_url") == ["a"]

    def test_복수형이_있으면_단수형은_무시된다(self):
        assert handler._as_list(["a"], "b", "audio_url") == ["a"]

    def test_둘_다_없으면_빈_리스트(self):
        assert handler._as_list(None, None, "audio_url") == []

    def test_원소는_문자열로_강제된다(self):
        assert handler._as_list([1, 2], None, "audio_url") == ["1", "2"]

    def test_리스트도_문자열도_아니면_ValueError(self):
        with pytest.raises(ValueError, match="audio_url"):
            handler._as_list(42, None, "audio_url")


class TestResolveBeamSize:
    def test_미지정이면_기본값(self):
        assert handler._resolve_beam_size({}, "job") == handler.DEFAULT_BEAM_SIZE

    def test_지정한_값을_그대로_쓴다(self):
        assert handler._resolve_beam_size({"beam_size": 3}, "job") == 3

    def test_상한을_넘으면_잘린다(self):
        # 이게 안 막히면 beam_size=500 한 번에 GPU 과금이 그대로 터진다.
        assert handler._resolve_beam_size({"beam_size": 500}, "job") == handler.MAX_BEAM_SIZE

    def test_상한_경계값은_통과한다(self):
        assert handler._resolve_beam_size(
            {"beam_size": handler.MAX_BEAM_SIZE}, "job"
        ) == handler.MAX_BEAM_SIZE

    def test_0이나_음수는_1로_올라간다(self):
        assert handler._resolve_beam_size({"beam_size": 0}, "job") == 1
        assert handler._resolve_beam_size({"beam_size": -5}, "job") == 1

    def test_문자열_숫자도_받는다(self):
        assert handler._resolve_beam_size({"beam_size": "3"}, "job") == 3

    def test_숫자가_아니면_ValueError(self):
        with pytest.raises(ValueError):
            handler._resolve_beam_size({"beam_size": "fast"}, "job")

    def test_legacy_batch_size는_5로_캡된다(self):
        assert handler._resolve_beam_size({"batch_size": 16}, "job") == 5

    def test_legacy_batch_size가_작으면_그대로(self):
        assert handler._resolve_beam_size({"batch_size": 2}, "job") == 2

    def test_beam_size가_batch_size를_이긴다(self):
        assert handler._resolve_beam_size({"beam_size": 7, "batch_size": 2}, "job") == 7


class TestResolveOutputFormats:
    def test_기본값은_text(self):
        assert handler._resolve_output_formats({}) == ["text"]

    def test_문자열_하나도_받는다(self):
        assert handler._resolve_output_formats({"output_formats": "srt"}) == ["srt"]

    def test_대문자는_소문자로_정규화된다(self):
        assert handler._resolve_output_formats({"output_formats": ["SRT", "VTT"]}) == ["srt", "vtt"]
