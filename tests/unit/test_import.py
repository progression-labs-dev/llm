"""Basic import tests to verify package structure."""


def test_import_package():
    """Test that the main package can be imported."""
    from progression_labs import llm

    assert llm.__version__ == "0.1.0"


def test_import_submodules():
    """Test that submodules can be imported."""
    from progression_labs.llm import eval, rag

    assert rag is not None
    assert eval is not None
