"""Versioned model store: every candidate is recorded, only winners keep a checkpoint."""

from __future__ import annotations

import json
import time

import torch

from .config import ModelConfig, TrainConfig, Workspace, from_dict, to_dict
from .model import GPT


class Registry:
    def __init__(self, ws: Workspace):
        self.ws = ws
        if ws.registry.exists():
            self.data = json.loads(ws.registry.read_text())
        else:
            self.data = {"champion": None, "counter": 0, "versions": {}}

    # -- bookkeeping -------------------------------------------------------
    def _save(self) -> None:
        self.ws.runs.mkdir(parents=True, exist_ok=True)
        tmp = self.ws.registry.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        tmp.replace(self.ws.registry)  # atomic: a crash never leaves a half-written registry

    @property
    def champion(self) -> str | None:
        return self.data["champion"]

    @property
    def versions(self) -> dict:
        return self.data["versions"]

    def next_id(self) -> str:
        self.data["counter"] += 1
        return f"v{self.data['counter']:04d}"

    def path(self, vid: str):
        return self.ws.models / f"{vid}.pt"

    # -- models ------------------------------------------------------------
    def record(self, vid: str, parent: str | None, action: str, report: dict,
               accepted: bool, model: GPT | None = None, tcfg: TrainConfig | None = None,
               note: str = "") -> None:
        entry = {
            "parent": parent,
            "action": action,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": report.get("score"),
            "report": {k: v for k, v in report.items() if k != "failures"},
            "status": "accepted" if accepted else "rejected",
            "note": note,
            "checkpoint": None,
        }
        if accepted and model is not None:
            self.ws.models.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_cfg": to_dict(model.cfg), "train_cfg": to_dict(tcfg or TrainConfig()),
                 "state": model.state_dict()},
                self.path(vid),
            )
            entry["checkpoint"] = self.path(vid).name
            entry["params"] = model.num_params()
            entry["layers"] = model.cfg.n_layer
        self.versions[vid] = entry
        if accepted:
            self.data["champion"] = vid
        self._save()

    def load(self, vid: str | None = None) -> tuple[GPT, TrainConfig]:
        vid = vid or self.champion
        if vid is None:
            raise FileNotFoundError("no model yet — run `python -m selfimprove init` first")
        ckpt = torch.load(self.path(vid), map_location="cpu", weights_only=True)
        model = GPT(from_dict(ModelConfig, ckpt["model_cfg"]))
        model.load_state_dict(ckpt["state"])
        model.eval()
        return model, from_dict(TrainConfig, ckpt["train_cfg"])

    def accepted_versions(self) -> list[str]:
        return [v for v, e in self.versions.items() if e["status"] == "accepted" and e.get("checkpoint")]

    def rollback(self, to: str | None = None, reason: str = "") -> str | None:
        """Make an earlier good version the champion again."""
        good = [v for v in self.accepted_versions() if self.path(v).exists()]
        if to is None:
            earlier = [v for v in good if v != self.champion]
            to = earlier[-1] if earlier else None
        if to is None or to not in good:
            return None
        if self.champion and self.champion in self.versions:
            self.versions[self.champion]["status"] = "rolled_back"
            self.versions[self.champion]["note"] += f" rolled back: {reason}".rstrip()
        self.versions[to]["status"] = "accepted"
        self.data["champion"] = to
        self._save()
        return to
