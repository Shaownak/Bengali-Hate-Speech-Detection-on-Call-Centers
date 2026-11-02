from pathlib import Path

from tools.lint_pii_logs import check_file, main


def write(tmp_path: Path, code: str) -> Path:
    p = tmp_path / "mod.py"
    p.write_text(code, encoding="utf-8")
    return p


def test_flags_content_keyword(tmp_path: Path) -> None:
    p = write(tmp_path, "log.info('x', text=seg)\n")
    assert any("'text'" in m for m in check_file(p))


def test_flags_content_reference(tmp_path: Path) -> None:
    p = write(tmp_path, "logger.warning(f'bad {segment.text_redacted}')\nprint(transcript)\n")
    msgs = check_file(p)
    assert any("text_redacted" in m for m in msgs)
    assert any("transcript" in m for m in msgs)


def test_flags_chained_structlog(tmp_path: Path) -> None:
    p = write(tmp_path, "structlog.get_logger().info('e', pcm=frame.pcm_s16le)\n")
    assert len(check_file(p)) >= 2


def test_allows_metadata(tmp_path: Path) -> None:
    p = write(tmp_path, "log.info('done', call_id=c, latency_ms=3, text_len=len(t))\n")
    assert check_file(p) == []


def test_method_names_are_not_content(tmp_path: Path) -> None:
    assert check_file(write(tmp_path, "print(path.read_text())\n")) == []
    assert check_file(write(tmp_path, "print(seg.text.strip())\n")) != []


def test_ignores_non_log_calls(tmp_path: Path) -> None:
    p = write(tmp_path, "store.save(text=t)\nredact(text)\n")
    assert check_file(p) == []


def test_main_exit_codes(tmp_path: Path) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "a.py").write_text("log.info('ok', call_id=1)\n", encoding="utf-8")
    assert main([str(good)]) == 0
    (good / "b.py").write_text("log.info('bad', transcript=t)\n", encoding="utf-8")
    assert main([str(good)]) == 1


def test_repo_is_clean() -> None:
    assert main([]) == 0
