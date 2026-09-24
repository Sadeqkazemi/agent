# agent — a local, self-improving language model

A small GPT-style language model that trains and runs **entirely offline** on your own machine.
It never calls an external API, and it has a self-improvement loop that:

1. **Examines itself:** loss, accuracy per skill, repetition in generated text, numerical health of the weights
2. **Diagnoses its faults:** weak skills, overfitting, underfitting, instability, repetition, plateaus
3. **Chooses a fix:** and learns which fixes have actually worked before
4. **Builds a new candidate version** (targeted training, training on its own verified answers, hyperparameter changes, or **growing the model**)
5. **Re-examines the candidate** and promotes it only if it is genuinely better and has not forgotten any skill; otherwise the candidate is discarded

If the current model gets corrupted (damaged checkpoint, NaN weights), the system automatically rolls back to the last healthy version.

## Installation

```bash
pip install -r requirements.txt      # only PyTorch
```

## Usage

```bash
python -m selfimprove init                 # train the tokenizer and the first model (v0001) from data/
python -m selfimprove improve --cycles 5   # run 5 self-improvement cycles
python -m selfimprove improve --cycles 0 --hours 8   # keep improving for 8 hours
python -m selfimprove status               # versions, strategy track record, recent mistakes
python -m selfimprove chat                 # talk to the current best version
python -m selfimprove teach "question" "answer"   # teach a new fact
python -m selfimprove doctor               # self-test the whole system and auto-repair
python -m selfimprove rollback --version v0003
python -m selfimprove eval
```

Inside `chat` you can also type: `/teach question => answer`

## What one improvement cycle looks like

```
[examine] champion v0004 (4 layers, 875,264 params)
  skills add=3%  reverse=17%  sort=93%  copy=27%  knowledge=100%
[diagnose]
  - overfitting (severity 1.00): val_loss exceeds train_loss by 10.44
  - weak_skills (severity 0.97): below target: add=3%, reverse=17%, copy=27%
[fix] targeted_training {'skills': ['add', 'reverse', 'copy']} for 'weak_skills'
[verify] candidate score 0.4376  skills add=0%  reverse=13%  sort=80%  copy=57%  knowledge=100%
[REJECTED] v0005: regression on sort: 93% -> 80%. champion is now v0004
```

## Faults and fixes

| Diagnosed fault | Symptom | Possible fixes |
|---|---|---|
| `weak_skills` | a skill is below the target accuracy (90%) | `targeted_training` (more practice on that skill), `self_training` |
| `overfitting` | large gap between val_loss and train_loss | `regularize` (more dropout and weight decay) |
| `underfitting` | high train_loss | `longer_training`, `grow` |
| `repetition` | generated text gets stuck in loops | `continue_training`, `regularize` |
| `instability` | infinite/NaN loss | `lower_lr` |
| `plateau` | several cycles in a row without progress | `grow`, `lower_lr`, `longer_training` |

- **`self_training`**: the model answers new questions, an automatic verifier keeps only the correct answers, and the model trains on its own correct work (the STaR method).
- **`grow` (upgrade)**: new layers are added whose outputs start at exactly zero, so the model's behaviour is unchanged at first. Nothing is forgotten, and the model gains capacity to learn more.
- **Strategy memory** (`runs/strategies.json`): how many times each fix was tried, how often it won, and its average score gain. The next fix is chosen with a UCB bandit: successful fixes are preferred, and rarely tried ones still get a chance.

## Data

- `data/corpus/*.txt` — raw text for learning the language. **This matters most for model quality.** The sample text is only a few kilobytes; the more text you add (books, articles, a Wikipedia dump downloaded once), the better the model writes.
- `data/knowledge.jsonl` — facts as `{"q": "...", "a": "..."}`. They are both examined and trained on.
- Verifiable skills live in `selfimprove/skills.py` (addition, word reversal, sorting, copying). To add a skill, write a function that generates question/answer pairs and register it in `builtin_skills`.

All outputs are stored in `runs/`: model versions, `registry.json`, `journal.jsonl` (a log of every cycle), and `strategies.json`.

## Model sizes

The network is a modern decoder-only transformer: rotary position embeddings (RoPE), RMSNorm,
SwiGLU feed-forward layers, weight tying and a KV cache for fast generation. Pick a size at init:

| `--size` | Parameters | Layers × width | Context | Vocab | Where it is practical |
|---|---|---|---|---|---|
| `tiny` (default) | ~1M | 4 × 128 | 128 | 512 | any CPU, minutes |
| `small` | ~12M | 6 × 384 | 256 | 4096 | CPU (hours) or any GPU (minutes) |
| `medium` | ~92M | 12 × 768 | 512 | 8192 | a GPU with 8 GB+ |
| `large` | ~320M | 24 × 1024 | 1024 | 16384 | a GPU with 24 GB+ |

```bash
python -m selfimprove init --force --size small
python -m selfimprove init --force --size medium --steps 10000   # any preset value can be overridden
```

On a GPU, training automatically uses bfloat16 mixed precision. Gradient accumulation (`--accum`)
gives large effective batches on small memory. The `grow` action keeps adding layers during
self-improvement (up to `max_layers`, default 48), so a model can also start small and grow.

**Bigger models need more text.** A 12M-parameter model needs at least tens of megabytes of text,
a 92M model hundreds of megabytes. With only the sample corpus, larger models just memorize it.
The tokenizer trainer is incremental and the encoded corpus is cached in `runs/`, so large corpora are fine.

Checkpoints from before this architecture (version 1) can't be loaded. Run `init --force` to rebuild.

## Configuration

Diagnosis and promotion thresholds (target accuracy, overfitting limit, maximum number of layers, etc.) are in `ImproveConfig` in `selfimprove/config.py` and can be overridden by creating `runs/settings.json`, for example:

```json
{"target_accuracy": 0.95, "max_layers": 16, "exam_size": 50}
```

## Limitations

- Even the `large` preset is far smaller than commercial assistants. With the sample data the model learns simple skills, but open-ended ChatGPT-like conversation **requires far more data (gigabytes of text) and strong hardware (a GPU)**.
- "Self-improvement" means improving the weights, hyperparameters and model size against measurable criteria. The model does not rewrite its own code — deliberately, because unsupervised code changes would make the system untrustworthy.
- The model only improves at things that can be checked automatically. For each new ability you want, add a verifiable skill or new facts.

## Tests

```bash
pip install pytest
python -m pytest -q
```
