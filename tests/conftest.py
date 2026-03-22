from pathlib import Path


SKIP_COLLECTION = {
    "test_halcon_call.py",
    "test_halcon_genimage.py",
    "test_halcon_insp.py",
    "test_machine_client.py",
    "test_touch_numpad.py",
}


def pytest_ignore_collect(collection_path: Path, config):
    return collection_path.name in SKIP_COLLECTION
