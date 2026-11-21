from pm_research import tracing


def test_span_is_noop_without_backends() -> None:
    with tracing.span("unit", as_type="tool", input={"a": 1}, foo="bar") as s:
        s.set("k", 1)
        s.set("obj", {"x": [1, 2]})
        s.output({"ok": True})
    assert not tracing.langfuse_enabled()
    tracing.flush()
