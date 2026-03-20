import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import_from_path():
    from qa_system_v2bis.synthetic_v2.production_runtime import import_from_path
    result = import_from_path("os.path:join")
    assert callable(result)

def test_import_dot_notation():
    from qa_system_v2bis.synthetic_v2.production_runtime import import_from_path
    result = import_from_path("os.path.join")
    assert callable(result)

def test_maybe_import_none():
    from qa_system_v2bis.synthetic_v2.production_runtime import maybe_import_callable
    assert maybe_import_callable(None) is None
    assert maybe_import_callable("") is None

def test_config_error_is_runtime_error():
    from qa_system_v2bis.synthetic_v2.production_runtime import RuntimeConfigError
    assert issubclass(RuntimeConfigError, RuntimeError)

def test_build_backend_missing():
    from qa_system_v2bis.synthetic_v2.production_runtime import build_backend_from_section, RuntimeConfigError
    import pytest
    with pytest.raises(RuntimeConfigError):
        build_backend_from_section(None, "test")
    with pytest.raises(RuntimeConfigError):
        build_backend_from_section({}, "test")

def test_build_backend_object_path():
    from qa_system_v2bis.synthetic_v2.production_runtime import build_backend_from_section
    result = build_backend_from_section({"backend_object_path": "os.path:sep"}, "test")
    assert result == os.sep