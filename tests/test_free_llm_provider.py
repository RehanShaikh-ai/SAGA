from types import SimpleNamespace
from unittest.mock import MagicMock

from saga.providers import FreeLLMProvider


def test_generate_returns_message_content():
    provider = FreeLLMProvider.__new__(FreeLLMProvider)
    client = MagicMock()
    provider.client = client
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))]
    )

    assert provider.generate([{"role": "user", "content": "Say hello"}]) == "Hello"


def test_list_models_returns_model_ids():
    provider = FreeLLMProvider.__new__(FreeLLMProvider)
    client = MagicMock()
    provider.client = client
    client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")]
    )

    assert provider.list_models() == ["model-a", "model-b"]


def test_stream_yields_only_non_empty_content():
    provider = FreeLLMProvider.__new__(FreeLLMProvider)
    client = MagicMock()
    provider.client = client
    client.chat.completions.create.return_value = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
    ]

    result = list(provider.stream([{"role": "user", "content": "Say hello"}]))

    assert result == ["Hel", "lo"]
    client.chat.completions.create.assert_called_once()
    