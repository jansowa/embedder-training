"""Compression of SPLADE models by distillation: student surgery, representation distillation, fidelity.

The package is independent of ``training/`` (the contrastive / ranking trainer) except for the
sparse-gradient decoder in ``training.backends.splade_fused_head``. Stages:

* ``distill.student``   build a smaller student from a teacher (layer / FFN pruning, low-rank vocabulary);
* ``distill.train_kd``  distil the teacher's token logits and pooled SPLADE vectors on unlabeled text;
* ``distill.fidelity``  measure how closely a student reproduces the teacher's vectors and how fast it is;
* the ranking stage (KL on teacher scores over training rows) stays in ``training.train``.
"""
