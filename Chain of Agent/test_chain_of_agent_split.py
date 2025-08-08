import os
import sys
import pytest

# Ensure the module path includes the directory of this file
sys.path.append(os.path.dirname(__file__))

from chain_of_agent import ChainOfAgents

def test_split_input_into_chunks_valid():
    c = ChainOfAgents(long_input="", context_window_size=2, model="m", api_key="k")
    chunks = c.split_input_into_chunks("a b c d", 2)
    assert chunks == ["a b", "c d"]


def test_split_input_into_chunks_invalid():
    c = ChainOfAgents(long_input="", context_window_size=1, model="m", api_key="k")
    with pytest.raises(ValueError):
        c.split_input_into_chunks("some text", 0)

