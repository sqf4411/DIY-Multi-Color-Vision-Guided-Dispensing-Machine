from core.machine_client import MachineClient


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.ok = True

    def json(self):
        return self._payload


class _DummySession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, *args, **kwargs):
        return _DummyResponse(self._payload)


def _build_client(payload):
    mc = MachineClient()
    mc.base_url = "http://127.0.0.1:7125"
    mc.connected = True
    mc.session = _DummySession(payload)
    return mc


def test_get_probe_status_string_zero_is_open():
    payload = {"result": {"status": {"probe": {"last_query": "0"}}}}
    mc = _build_client(payload)
    assert mc.get_probe_status() == "open"


def test_get_probe_status_string_one_is_triggered():
    payload = {"result": {"status": {"probe": {"last_query": "1"}}}}
    mc = _build_client(payload)
    assert mc.get_probe_status() == "triggered"


def test_get_output_pin_status_string_numeric_is_supported():
    payload = {"result": {"status": {"output_pin valve": {"value": "1.0"}}}}
    mc = _build_client(payload)
    assert mc.get_output_pin_status("valve") is True
