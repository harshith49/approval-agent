import pytest
from app.llm import MockModel, create_model


def test_mock_needs_no_key(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    assert isinstance(create_model('mock'), MockModel)
    with pytest.raises(ValueError, match='GEMINI_API_KEY'):
        create_model('gemini')
    with pytest.raises(ValueError, match='Unknown'):
        create_model('invalid')


def test_optional_integrations_bind_tools_without_network(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-placeholder-not-a-real-key')
    monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', 'false')
    for mode in ('ollama', 'gemini'):
        model = create_model(mode)
        assert model.kwargs['tools']
