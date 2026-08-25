"""URL 가드·크기 상한·병렬 다운로드·per-file 격리 테스트."""

import base64
import os
import time

import pytest

import handler


def _addrinfo(ip: str):
    return [(2, 1, 6, "", (ip, 443))]


@pytest.fixture
def resolve_to(monkeypatch):
    """getaddrinfo를 고정한다 — 테스트가 실제 DNS를 타지 않게."""

    def _set(ip: str):
        monkeypatch.setattr(handler.socket, "getaddrinfo", lambda *a, **k: _addrinfo(ip))

    return _set


class TestAssertPublicUrl:
    def test_file_스킴은_막힌다(self):
        # urllib은 file://를 그대로 열어준다. 막지 않으면 /etc/passwd가 읽힌다.
        with pytest.raises(ValueError, match="unsupported URL scheme"):
            handler.assert_public_url("file:///etc/passwd")

    def test_ftp_스킴도_막힌다(self):
        with pytest.raises(ValueError, match="unsupported URL scheme"):
            handler.assert_public_url("ftp://example.com/a.wav")

    def test_host가_없으면_막힌다(self):
        with pytest.raises(ValueError, match="no host"):
            handler.assert_public_url("https:///a.wav")

    def test_클라우드_메타데이터_주소는_막힌다(self, resolve_to):
        resolve_to("169.254.169.254")
        with pytest.raises(ValueError, match="non-public address"):
            handler.assert_public_url("http://169.254.169.254/latest/meta-data/")

    def test_루프백은_막힌다(self, resolve_to):
        resolve_to("127.0.0.1")
        with pytest.raises(ValueError, match="non-public address"):
            handler.assert_public_url("http://localhost/a.wav")

    def test_사설대역은_막힌다(self, resolve_to):
        resolve_to("10.0.0.5")
        with pytest.raises(ValueError, match="non-public address"):
            handler.assert_public_url("https://internal.example.com/a.wav")

    def test_DNS_리바인딩_시도도_해석된_IP로_막힌다(self, resolve_to):
        # 공개 도메인처럼 보여도 A레코드가 사설이면 막혀야 한다.
        resolve_to("192.168.0.10")
        with pytest.raises(ValueError, match="non-public address"):
            handler.assert_public_url("https://totally-public.example.com/a.wav")

    def test_공개_주소는_통과한다(self, resolve_to):
        resolve_to("93.184.216.34")
        handler.assert_public_url("https://example.com/a.wav")

    def test_이름_해석_실패는_ValueError(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise handler.socket.gaierror("nope")

        monkeypatch.setattr(handler.socket, "getaddrinfo", _boom)
        with pytest.raises(ValueError, match="cannot resolve host"):
            handler.assert_public_url("https://nx.example.com/a.wav")

    def test_옵트인하면_사설대역도_통과한다(self, monkeypatch, resolve_to):
        resolve_to("10.0.0.5")
        monkeypatch.setattr(handler, "ALLOW_PRIVATE_URLS", True)
        handler.assert_public_url("https://internal.example.com/a.wav")


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, _size):
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    def __init__(self, chunks):
        self._chunks = chunks

    def open(self, _url, timeout=None):
        return _FakeResponse(self._chunks)


class TestDownloadAudio:
    @pytest.fixture(autouse=True)
    def _allow_host(self, monkeypatch):
        monkeypatch.setattr(handler, "assert_public_url", lambda url: None)

    def test_상한_안이면_전부_기록된다(self, monkeypatch, tmp_path):
        monkeypatch.setattr(handler, "_url_opener", _FakeOpener([b"ab", b"cd"]))
        target = tmp_path / "out.wav"
        with open(target, "wb") as fp:
            handler.download_audio("https://example.com/a.wav", fp)
        assert target.read_bytes() == b"abcd"

    def test_상한을_넘으면_중단된다(self, monkeypatch, tmp_path):
        # 상한이 없으면 거대한 URL 하나가 컨테이너 디스크를 채우면서 GPU 시간을 태운다.
        monkeypatch.setattr(handler, "MAX_DOWNLOAD_BYTES", 3)
        monkeypatch.setattr(handler, "MAX_DOWNLOAD_MB", 1)
        monkeypatch.setattr(handler, "_url_opener", _FakeOpener([b"ab", b"cd"]))
        target = tmp_path / "out.wav"
        with pytest.raises(ValueError, match="exceeds"):
            with open(target, "wb") as fp:
                handler.download_audio("https://example.com/a.wav", fp)

    def test_상한_정확히_맞으면_통과한다(self, monkeypatch, tmp_path):
        monkeypatch.setattr(handler, "MAX_DOWNLOAD_BYTES", 4)
        monkeypatch.setattr(handler, "_url_opener", _FakeOpener([b"ab", b"cd"]))
        target = tmp_path / "out.wav"
        with open(target, "wb") as fp:
            handler.download_audio("https://example.com/a.wav", fp)
        assert target.read_bytes() == b"abcd"


class TestSuffixFor:
    def test_경로_확장자를_쓴다(self):
        assert handler._suffix_for("https://e.com/a/b.opus") == ".opus"

    def test_쿼리스트링은_무시된다(self):
        assert handler._suffix_for("https://e.com/a.mp3?token=x.y") == ".mp3"

    def test_확장자가_없으면_wav로_떨어진다(self):
        assert handler._suffix_for("https://e.com/audio") == ".wav"

    def test_이상한_확장자는_wav로_떨어진다(self):
        assert handler._suffix_for("https://e.com/a.this-is-not-an-extension") == ".wav"


def _cleanup(items):
    for item in items:
        if "path" in item:
            os.unlink(item["path"])


class TestMaterializeInputs:
    def test_base64_먼저_URL_다음_순서가_유지된다(self, monkeypatch):
        monkeypatch.setattr(handler, "_materialize_url", lambda url: {"error": url})
        items = handler.materialize_inputs(
            [base64.b64encode(b"one").decode(), base64.b64encode(b"two").decode()],
            ["u1", "u2"],
        )
        try:
            assert [("path" in i) for i in items] == [True, True, False, False]
            assert [items[2]["error"], items[3]["error"]] == ["u1", "u2"]
        finally:
            _cleanup(items)

    def test_깨진_base64는_그_슬롯만_실패한다(self):
        items = handler.materialize_inputs(["a", base64.b64encode(b"ok").decode()], [])
        try:
            assert "invalid base64" in items[0]["error"]
            assert "path" in items[1]
        finally:
            _cleanup(items)

    def test_다운로드_실패는_그_슬롯만_실패한다(self, monkeypatch):
        # 예전에는 URL 하나가 죽으면 job 전체가 죽어서 이미 태운 GPU 시간까지 날아갔다.
        def _fake_download(url, fp):
            if "bad" in url:
                raise OSError("404")
            fp.write(b"ok")

        monkeypatch.setattr(handler, "download_audio", _fake_download)
        items = handler.materialize_inputs([], ["https://e.com/bad.wav", "https://e.com/ok.wav"])
        try:
            assert "download failed" in items[0]["error"]
            assert "path" in items[1]
        finally:
            _cleanup(items)

    def test_실패한_다운로드의_임시파일은_남지_않는다(self, monkeypatch):
        created = []

        def _fake_download(url, fp):
            created.append(fp.name)
            raise OSError("boom")

        monkeypatch.setattr(handler, "download_audio", _fake_download)
        items = handler.materialize_inputs([], ["https://e.com/a.wav"])
        assert "error" in items[0]
        assert created and not os.path.exists(created[0])

    def test_URL은_동시에_받아온다(self, monkeypatch):
        # 직렬이면 4 x 0.15s = 0.6s. 병렬이면 0.15s 언저리에서 끝나야 한다.
        monkeypatch.setattr(handler, "DOWNLOAD_MAX_THREADS", 8)

        def _slow_download(url, fp):
            time.sleep(0.15)
            fp.write(b"ok")

        monkeypatch.setattr(handler, "download_audio", _slow_download)
        urls = [f"https://e.com/{i}.wav" for i in range(4)]

        started = time.perf_counter()
        items = handler.materialize_inputs([], urls)
        elapsed = time.perf_counter() - started

        try:
            assert all("path" in item for item in items)
            assert elapsed < 0.4, f"직렬 다운로드로 보인다 ({elapsed:.2f}s)"
        finally:
            _cleanup(items)

    def test_병렬로_받아도_입력_순서가_유지된다(self, monkeypatch):
        monkeypatch.setattr(handler, "DOWNLOAD_MAX_THREADS", 8)

        def _staggered_download(url, fp):
            # 먼저 요청한 것이 늦게 끝나도 순서가 뒤집히면 안 된다.
            time.sleep(0.05 * (3 - int(url[-5])))
            fp.write(url[-5].encode())

        monkeypatch.setattr(handler, "download_audio", _staggered_download)
        urls = [f"https://e.com/{i}.wav" for i in range(4)]
        items = handler.materialize_inputs([], urls)
        try:
            contents = [open(item["path"], "rb").read() for item in items]
            assert contents == [b"0", b"1", b"2", b"3"]
        finally:
            _cleanup(items)


class TestMergeResults:
    def test_에러_슬롯_사이로_결과가_정렬된다(self):
        items = [{"error": "e1"}, {"path": "/tmp/a"}, {"error": "e2"}, {"path": "/tmp/b"}]
        merged = handler.merge_results(items, [{"transcription": "A"}, {"transcription": "B"}])
        assert [m.get("transcription", m.get("error")) for m in merged] == ["e1", "A", "e2", "B"]

    def test_에러_슬롯에도_inference_time이_붙는다(self):
        merged = handler.merge_results([{"error": "e"}], [])
        assert merged[0] == {"error": "e", "inference_time": 0.0}

    def test_전부_실패해도_슬롯_수가_유지된다(self):
        items = [{"error": "e1"}, {"error": "e2"}]
        assert len(handler.merge_results(items, [])) == 2
