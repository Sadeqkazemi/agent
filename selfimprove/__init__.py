"""Self-improving local language model.

A small GPT trained fully offline, plus a loop that evaluates the model,
diagnoses its weaknesses, applies a fix, and promotes the result only when
it is measurably better than the current champion.
"""

__version__ = "0.1.0"
