"""LSTM word-level language model (PyTorch).

Embedding -> dropout -> LSTM -> projection -> (tied) softmax over the vocabulary.
Trained with cross-entropy (teacher forcing) and Adam; validation perplexity
drives learning-rate decay and early stopping. Uses the same ``Vocab`` and
the same <s> ... </s> framing as the trigram model so perplexities are comparable.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NON_SUGGESTIBLE, SEED
from .vocab import Vocab


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 256, hidden: int = 512,
                 layers: int = 1, dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, emb=emb, hidden=hidden,
                        layers=layers, dropout=dropout, pad_id=pad_id)
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, emb)
        self.out = nn.Linear(emb, vocab_size)
        self.out.weight = self.embedding.weight   # weight tying
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor, state=None):
        h = self.drop(self.embedding(x))
        h, state = self.lstm(h, state)
        h = self.drop(self.proj(self.drop(h)))
        return self.out(h), state


def _frame(sentences: list[list[int]], vocab: Vocab, max_len: int | None) -> list[list[int]]:
    out = []
    for s in sentences:
        if max_len:
            s = s[:max_len]
        out.append([vocab.bos] + s + [vocab.eos])
    return out


def _batches(seqs: list[list[int]], max_tokens: int, pad: int, shuffle: bool, rng: random.Random):
    """Length-bucketed batches of at most ``max_tokens`` padded tokens."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(seqs[i])
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    for b in batches:
        L = -(-max(len(seqs[i]) for i in b) // 8) * 8
        arr = np.full((len(b), L), pad, dtype=np.int64)
        for r, i in enumerate(b):
            arr[r, : len(seqs[i])] = seqs[i]
        yield b, torch.from_numpy(arr)


@torch.no_grad()
def evaluate_nll(model: LSTMLM, seqs: list[list[int]], pad: int, device, max_tokens=8000):
    model.eval()
    total, n = 0.0, 0
    for _, batch in _batches(seqs, max_tokens, pad, False, random.Random(0)):
        batch = batch.to(device)
        logits, _ = model(batch[:, :-1])
        tgt = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=pad, reduction="sum")
        total += loss.item()
        n += (tgt != pad).sum().item()
    return total / n, n


def train_lstm(train: list[list[int]], val: list[list[int]], vocab: Vocab, *,
               emb=256, hidden=512, layers=1, dropout=0.3, lr=2e-3, epochs=8,
               max_tokens=4000, max_len=60, patience=2, clip=1.0, log=print,
               time_budget_s: float | None = None) -> tuple[LSTMLM, list[dict]]:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device()
    model = LSTMLM(len(vocab), emb, hidden, layers, dropout, vocab.pad).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0)
    tr, va = _frame(train, vocab, max_len), _frame(val, vocab, None)
    rng = random.Random(SEED)
    history, best, best_state, bad = [], float("inf"), None, 0
    t0 = time.time()
    log(f"LSTM LM on {device}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
        f"{len(tr)} train / {len(va)} val sentences, vocab {len(vocab)}")
    for ep in range(1, epochs + 1):
        model.train()
        tot, cnt, te = 0.0, 0, time.time()
        for bi, (_, batch) in enumerate(_batches(tr, max_tokens, vocab.pad, True, rng)):
            batch = batch.to(device)
            logits, _ = model(batch[:, :-1])
            tgt = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                   ignore_index=vocab.pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            n = (tgt != vocab.pad).sum().item()
            tot += loss.item() * n; cnt += n
            if device.type == "mps" and bi % 100 == 99:
                torch.mps.empty_cache()
            if time_budget_s and bi % 50 == 49 and time.time() - t0 > time_budget_s:
                break
        val_nll, _ = evaluate_nll(model, va, vocab.pad, device)
        sched.step(val_nll)
        rec = dict(epoch=ep, train_ppl=math.exp(tot / cnt), val_ppl=math.exp(val_nll),
                   lr=opt.param_groups[0]["lr"], seconds=time.time() - te)
        history.append(rec)
        log(f"  epoch {ep}: train ppl {rec['train_ppl']:.1f}  val ppl {rec['val_ppl']:.1f}  "
            f"({rec['seconds']:.0f}s)")
        if val_nll < best - 1e-4:
            best, bad = val_nll, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                log("  early stopping")
                break
        if time_budget_s and time.time() - t0 > time_budget_s:
            log("  time budget reached")
            break
    model.load_state_dict(best_state)
    return model, history


class LSTMPredictor:
    """Inference wrapper: perplexity, next-word suggestions and top-k accuracy."""

    def __init__(self, model: LSTMLM, vocab: Vocab, device=None):
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        mask = torch.zeros(len(vocab))
        for i, w in enumerate(vocab.itos):
            if w in NON_SUGGESTIBLE:
                mask[i] = float("-inf")
        self._mask = mask.to(self.device)

    @torch.no_grad()
    def perplexity(self, sentences: list[list[int]]) -> float:
        seqs = _frame(sentences, self.vocab, None)
        nll, _ = evaluate_nll(self.model, seqs, self.vocab.pad, self.device)
        return math.exp(nll)

    @torch.no_grad()
    def token_logprobs(self, ids: list[int]) -> list[float]:
        """Natural-log probability of every word in ``ids`` followed by ``</s>``."""
        seq = torch.tensor([[self.vocab.bos] + list(ids) + [self.vocab.eos]], device=self.device)
        logits, _ = self.model(seq[:, :-1])
        logp = F.log_softmax(logits[0], dim=-1)
        return logp.gather(1, seq[0, 1:].unsqueeze(1)).squeeze(1).tolist()

    @torch.no_grad()
    def predict_next(self, context_ids: list[int], k: int = 5) -> list[tuple[str, float]]:
        x = torch.tensor([[self.vocab.bos] + list(context_ids)], device=self.device)
        logits, _ = self.model(x)
        logp = F.log_softmax(logits[0, -1], dim=-1)          # full-vocabulary probabilities
        top = torch.topk(logp + self._mask, k)
        return [(self.vocab.itos[i], math.exp(logp[i].item())) for i in top.indices.tolist()]

    @torch.no_grad()
    def topk_accuracy(self, sentences: list[list[int]], ks=(1, 5), skip: set[int] = frozenset()):
        """Fraction of positions whose true word is in the model's top-k.

        Positions whose target is in ``skip`` (<unk>, <num>, </s>) are ignored,
        identical to the protocol used for the N-gram model.
        """
        seqs = [[self.vocab.bos] + s + [self.vocab.eos] for s in sentences]
        hits = {k: 0 for k in ks}
        total, kmax = 0, max(ks)
        for _, batch in _batches(seqs, 4000, self.vocab.pad, False, random.Random(0)):
            batch = batch.to(self.device)
            logits, _ = self.model(batch[:, :-1])
            tgt = batch[:, 1:]
            top = torch.topk(logits + self._mask, kmax, dim=-1).indices   # (B, T, kmax)
            valid = tgt != self.vocab.pad
            for sid in skip:
                valid &= tgt != sid
            match = top == tgt.unsqueeze(-1)
            for k in ks:
                hits[k] += (match[..., :k].any(-1) & valid).sum().item()
            total += valid.sum().item()
        return {k: hits[k] / total for k in ks}, total

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        torch.save({"cfg": self.model.cfg, "state": self.model.state_dict(),
                    "itos": self.vocab.itos}, path)

    @classmethod
    def load(cls, path: Path) -> "LSTMPredictor":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        vocab = Vocab(ck["itos"])
        model = LSTMLM(**ck["cfg"])
        model.load_state_dict(ck["state"])
        return cls(model, vocab, device=torch.device("cpu"))
