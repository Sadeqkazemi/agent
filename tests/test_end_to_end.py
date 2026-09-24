import shutil
from pathlib import Path

from selfimprove.config import ModelConfig, TrainConfig, Workspace
from selfimprove.loop import bootstrap, doctor, improve_cycle, read_journal
from selfimprove.registry import Registry

DATA = Path(__file__).resolve().parents[1] / "data"


def test_bootstrap_and_one_cycle(tmp_path):
    shutil.copytree(DATA, tmp_path / "data")
    ws = Workspace(tmp_path)
    quiet = lambda *a: None
    mcfg = ModelConfig(n_layer=1, n_embd=32, n_head=2, vocab_size=300, block_size=48)
    bootstrap(ws, mcfg, TrainConfig(steps=5, batch_size=4), log=quiet)
    assert Registry(ws).champion == "v0001"

    entry = improve_cycle(ws, log=quiet)
    reg = Registry(ws)
    assert entry["version"] == "v0002"
    assert reg.versions["v0002"]["status"] in ("accepted", "rejected")
    assert reg.champion == ("v0002" if entry["accepted"] else "v0001")
    assert len(read_journal(ws)) == 2
    assert doctor(ws, log=quiet)
