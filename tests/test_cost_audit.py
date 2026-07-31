from forgeos.cost_audit import CostAuditor


def test_audit_file_decodes_utf8_independently_of_windows_locale(tmp_path):
    source = tmp_path / "unicode_source.py"
    source.write_bytes("# café — source\n".encode("utf-8"))

    auditor = CostAuditor()
    assert auditor.audit_file(source) == []
    assert auditor.issues == []


def test_audit_file_replaces_malformed_bytes_instead_of_crashing(tmp_path):
    source = tmp_path / "malformed_source.py"
    source.write_bytes(b"# valid python comment\n\x9d\n")

    auditor = CostAuditor()
    assert auditor.audit_file(source) == []


def test_audit_directory_skips_generated_and_vendored_python(tmp_path):
    (tmp_path / "app.py").write_text("client.chat.completions.create('x')\n", encoding="utf-8")
    for dirname in ("vendor", ".venv", ".forgeos"):
        path = tmp_path / dirname
        path.mkdir()
        (path / "third_party.py").write_text(
            "client.chat.completions.create('x')\n", encoding="utf-8"
        )

    result = CostAuditor().audit_directory(tmp_path)

    assert result["files_audited"] == 1
    assert result["total_issues"] == 1
