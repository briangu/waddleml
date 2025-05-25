import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from waddle.server import WaddleServer, LogFileEventHandler

class DummyServer(WaddleServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ingested = []

    def _ingest_log_entry(self, log_entry):
        self.ingested.append(log_entry)


def create_event(path):
    class Event:
        def __init__(self, src_path):
            self.src_path = str(src_path)
            self.is_directory = False
    return Event(path)


def test_json_file_ingested_and_removed(tmp_path):
    server = DummyServer(db_root=tmp_path, log_root=tmp_path)
    handler = LogFileEventHandler(server)

    data = {"a": 1}
    file_path = tmp_path / "log.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    handler.on_created(create_event(file_path))

    assert server.ingested == [data]
    assert not file_path.exists()


def test_non_json_file_ignored(tmp_path):
    server = DummyServer(db_root=tmp_path, log_root=tmp_path)
    handler = LogFileEventHandler(server)

    file_path = tmp_path / "log.txt"
    file_path.write_text("ignored")

    handler.on_created(create_event(file_path))

    assert server.ingested == []
    assert file_path.exists()
