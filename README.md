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

## Configuration

A bigger model (if you have a GPU or more time):

```bash
python -m selfimprove init --force --layers 8 --embd 384 --heads 6 --vocab 4096 --block 256 --steps 5000
```

Diagnosis and promotion thresholds (target accuracy, overfitting limit, maximum number of layers, etc.) are in `ImproveConfig` in `selfimprove/config.py` and can be overridden by creating `runs/settings.json`, for example:

```json
{"target_accuracy": 0.95, "max_layers": 16, "exam_size": 50}
```

## Limitations

- This is a small model that also runs on CPU. With the sample data it learns simple skills, but open-ended ChatGPT-like conversation **requires far more data (gigabytes of text) and strong hardware (a GPU)**.
- "Self-improvement" means improving the weights, hyperparameters and model size against measurable criteria. The model does not rewrite its own code — deliberately, because unsupervised code changes would make the system untrustworthy.
- The model only improves at things that can be checked automatically. For each new ability you want, add a verifiable skill or new facts.

## Tests

```bash
pip install pytest
python -m pytest -q
```
