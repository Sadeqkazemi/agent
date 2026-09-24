import random

import pytest
import torch

from selfimprove.config import ImproveConfig, ModelConfig, TrainConfig, Workspace
from selfimprove.diagnosis import diagnose
from selfimprove.loop import judge, self_repair
from selfimprove.model import GPT
from selfimprove.planner import StrategyMemory, choose
from selfimprove.registry import Registry
from selfimprove.skills import build_exam, builtin_skills, sample_training
from selfimprove.tokenizer import BPETokenizer

TEXT = "سلام دنیا! Hello world 123. " * 20 + "reverse abc\nA: cba\n"


def test_tokenizer_roundtrip_and_digits():
    tok = BPETokenizer.train(TEXT, 300)
    for s in (TEXT, "متن تازه که دیده نشده ✓", "45+67=112"):
        assert tok.decode(tok.encode(s)) == s
    # digits are never merged, so arithmetic stays learnable
    assert len(tok.encode("12345")) == 5


def test_tokenizer_save_load(tmp_path):
    tok = BPETokenizer.train(TEXT, 300)
    tok.save(tmp_path / "t.json")
    assert BPETokenizer.load(tmp_path / "t.json").encode(TEXT) == tok.encode(TEXT)


def tiny_model(layers=2):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=300, block_size=32, n_layer=layers, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_growth_preserves_function():
    m = tiny_model()
    x = torch.randint(0, 300, (2, 16))
    big = m.grown(2).eval()
    assert big.cfg.n_layer == 4
    with torch.no_grad():
        assert torch.allclose(m(x)[0], big(x)[0], atol=1e-5)


def test_generate_stops():
    m = tiny_model()
    out = m.generate(torch.zeros(1, 3, dtype=torch.long), 10, stop=lambda new: len(new) >= 4)
    assert out.shape[1] == 7


def test_exam_is_held_out_from_training():
    skills = builtin_skills()
    exam = build_exam(skills, 20)
    for name, skill in skills.items():
        qs = {q for q, _ in exam[name]}
        assert len(qs) == 20
        train = sample_training(skill, random.Random(1), 200, qs)
        assert not qs & {q for q, _ in train}
        assert all(skill.check(a, a) for _, a in exam[name])


def report(skills, val=2.0, train=1.9, rep=0.1, healthy=True, score=0.5):
    return {"healthy": healthy, "skills": skills, "val_loss": val, "train_loss": train,
            "repetition": rep, "score": score}


def test_diagnosis_finds_problems():
    icfg = ImproveConfig()
    kinds = {i.kind for i in diagnose(report({"add": 0.2, "copy": 1.0}, val=3.0, train=1.0), icfg, [], 4)}
    assert {"weak_skills", "overfitting"} <= kinds
    stalled = [{"accepted": False}] * icfg.plateau_cycles
    assert "plateau" in {i.kind for i in diagnose(report({"add": 1.0}), icfg, stalled, 4)}
    assert "instability" in {i.kind for i in diagnose(report({"add": 1.0}), icfg, [{"diverged": True}], 4)}
    assert [i.kind for i in diagnose(report({"add": 1.0}, val=1.0, train=1.0), icfg, [], 4)] == ["polish"]


def test_judge_blocks_regressions_and_noise():
    icfg = ImproveConfig()
    old = report({"a": 0.8, "b": 0.5}, score=0.5)
    assert judge(old, report({"a": 0.9, "b": 0.7}, score=0.6), icfg)[0]
    assert not judge(old, report({"a": 0.5, "b": 1.0}, score=0.7), icfg)[0]  # forgot skill a
    assert not judge(old, report({"a": 0.8, "b": 0.5}, score=0.5001), icfg)[0]
    assert not judge(old, report({"a": 1, "b": 1}, healthy=False, score=0.9), icfg)[0]


def test_planner_learns_from_outcomes(tmp_path):
    ws = Workspace(tmp_path)
    ws.ensure()
    mem = StrategyMemory(ws)
    issues = diagnose(report({"add": 0.2}), ImproveConfig(), [], 4)
    for _ in range(5):
        mem.update("targeted_training", -0.05, False)
        mem.update("self_training", 0.05, True)
    assert choose(issues, mem, 0.05)[1] == "self_training"


def test_registry_rollback_and_self_repair(tmp_path):
    ws = Workspace(tmp_path)
    ws.ensure()
    reg = Registry(ws)
    r = report({"a": 1.0})
    for _ in range(2):
        reg.record(reg.next_id(), reg.champion, "x", r, True, tiny_model(), TrainConfig())
    assert reg.champion == "v0002"
    reg.path("v0002").write_bytes(b"corrupted")
    model, _ = self_repair(ws, Registry(ws), log=lambda *a: None)
    assert Registry(ws).champion == "v0001"
    assert model.is_finite()


def test_self_repair_gives_up_without_healthy_version(tmp_path):
    ws = Workspace(tmp_path)
    ws.ensure()
    reg = Registry(ws)
    reg.record(reg.next_id(), None, "x", report({"a": 1.0}), True, tiny_model(), TrainConfig())
    reg.path("v0001").unlink()
    with pytest.raises(RuntimeError):
        self_repair(ws, Registry(ws), log=lambda *a: None)
