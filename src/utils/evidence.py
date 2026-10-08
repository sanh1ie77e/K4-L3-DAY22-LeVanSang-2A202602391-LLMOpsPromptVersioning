"""Lưu stdout thực tế bằng UTF-8, đồng thời giữ output trên terminal."""
import sys
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        self.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_logs(*filenames):
    """Ghi log của lần chạy hiện tại vào evidence/."""
    folder = Path(__file__).resolve().parents[2] / "evidence"
    folder.mkdir(exist_ok=True)
    with ExitStack() as stack:
        files = [stack.enter_context((folder / name).open("w", encoding="utf-8"))
                 for name in filenames]
        with redirect_stdout(_Tee(sys.stdout, *files)):
            yield
