"""자막 타임스탬프·포맷 테스트. 오프바이원이 조용히 나기 쉬운 자리라 값을 못 박는다."""

import handler


class TestFormatTimestamp:
    def test_영초는_0으로_표시된다(self):
        assert handler._format_timestamp(0, ",") == "00:00:00,000"

    def test_밀리초가_반올림된다(self):
        assert handler._format_timestamp(1.2345, ",") == "00:00:01,234"
        assert handler._format_timestamp(1.2355, ",") == "00:00:01,236"

    def test_분_초_경계(self):
        assert handler._format_timestamp(59.999, ",") == "00:00:59,999"
        assert handler._format_timestamp(60.0, ",") == "00:01:00,000"

    def test_시간_단위까지_올라간다(self):
        assert handler._format_timestamp(3661.5, ",") == "01:01:01,500"

    def test_한시간_넘는_통화도_자릿수가_유지된다(self):
        assert handler._format_timestamp(36000, ",") == "10:00:00,000"

    def test_음수는_0으로_막힌다(self):
        assert handler._format_timestamp(-1.5, ",") == "00:00:00,000"

    def test_vtt는_소수점을_점으로_쓴다(self):
        assert handler._format_timestamp(1.5, ".") == "00:00:01.500"


SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": " 첫 번째 "},
    {"start": 2.5, "end": 4.0, "text": "두 번째"},
]


class TestBuildSrt:
    def test_인덱스는_1부터_시작한다(self):
        assert handler.build_srt(SEGMENTS).startswith("1\n")

    def test_전체_블록_형태(self):
        assert handler.build_srt(SEGMENTS) == (
            "1\n00:00:00,000 --> 00:00:02,500\n첫 번째\n\n"
            "2\n00:00:02,500 --> 00:00:04,000\n두 번째\n\n"
        )

    def test_빈_세그먼트는_빈_문자열(self):
        assert handler.build_srt([]) == ""


class TestBuildVtt:
    def test_WEBVTT_헤더로_시작한다(self):
        assert handler.build_vtt(SEGMENTS).startswith("WEBVTT\n\n")

    def test_전체_블록_형태(self):
        assert handler.build_vtt(SEGMENTS) == (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.500\n첫 번째\n\n"
            "00:00:02.500 --> 00:00:04.000\n두 번째\n\n"
        )

    def test_빈_세그먼트도_헤더는_남는다(self):
        assert handler.build_vtt([]) == "WEBVTT\n\n"


class TestAddSubtitleOutputs:
    def test_요청_안하면_키가_안_붙는다(self):
        results = [{"segments": SEGMENTS}]
        handler._add_subtitle_outputs(results, ["text"])
        assert "srt" not in results[0] and "vtt" not in results[0]

    def test_에러_슬롯은_건너뛴다(self):
        results = [{"error": "download failed", "inference_time": 0.0}]
        handler._add_subtitle_outputs(results, ["srt", "vtt"])
        assert "srt" not in results[0] and "vtt" not in results[0]

    def test_성공_슬롯에만_붙는다(self):
        results = [{"error": "boom"}, {"segments": SEGMENTS}]
        handler._add_subtitle_outputs(results, ["srt"])
        assert "srt" not in results[0]
        assert results[1]["srt"].startswith("1\n")
