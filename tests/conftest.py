from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch):
    """Keep library tests independent of model/API downloads and sockets."""

    def denied_connection(*args, **kwargs):
        raise AssertionError("tests must not open real network connections")

    monkeypatch.setattr(socket.socket, "connect", denied_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", denied_connection)
