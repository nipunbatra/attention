"""Shared, inspectable utilities for the word-level next-token notebooks.

The notebooks keep the central model equations in their own code cells.  This
module holds the repeatable data, training, evaluation, and artifact plumbing
used by both the notebooks and the command-line experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DATASET_REPO = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_LICENSE = "CDLA-Sharing-1.0"
DATASET_API = "https://datasets-server.huggingface.co/rows"
HUB_API = "https://huggingface.co/api/datasets/roneneldan/TinyStories"
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", re.IGNORECASE)

PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)

PROFILE_SPECS: dict[str, dict[str, int]] = {
    "smoke": {"stories": 480, "max_vocab": 1_500, "min_freq": 2, "offset_seed": 1701},
    "classroom": {"stories": 3_000, "max_vocab": 3_000, "min_freq": 2, "offset_seed": 1701},
    "dgx": {"stories": 6_000, "max_vocab": 4_000, "min_freq": 2, "offset_seed": 1701},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_url(url: str, timeout: int = 60, retries: int = 8) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dl-teaching-wordlm/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception as error:  # pragma: no cover - exercised only on network failures
            last_error = error
            if attempt + 1 < retries:
                retry_after = 0.0
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    retry_after = float(error.headers.get("Retry-After", 0) or 0)
                time.sleep(max(retry_after, min(30.0, 1.5 * 2**attempt)))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def current_dataset_revision() -> str:
    return str(_read_json_url(HUB_API)["sha"])


def deterministic_batch_offsets(
    stories: int,
    seed: int,
    total_rows: int = 2_119_719,
    page_size: int = 100,
) -> list[int]:
    """Choose reproducible, non-overlapping row pages across the full train split."""
    pages = math.ceil(stories / page_size)
    population = range(0, total_rows - page_size + 1, page_size)
    offsets = random.Random(seed).sample(list(population), pages)
    return sorted(offsets)


def fetch_bounded_tinystories(
    output_dir: Path | str,
    profile: str = "smoke",
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch a deterministic row slice without executing dataset repository code.

    The Dataset Viewer does not accept a historical revision parameter.  We
    therefore fail closed unless the Hub repository's current commit equals
    the recorded revision, and we record a SHA-256 checksum of the exact rows.
    """
    if profile not in PROFILE_SPECS:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILE_SPECS)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "stories.jsonl"
    manifest_path = output_dir / "source_manifest.json"
    if raw_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        actual = sha256_file(raw_path)
        if actual != manifest["raw_sha256"]:
            raise RuntimeError(f"Cached data checksum mismatch: {actual} != {manifest['raw_sha256']}")
        if manifest["dataset_revision"] != DATASET_REVISION:
            raise RuntimeError("Cached data came from a different TinyStories revision")
        return manifest

    revision = current_dataset_revision()
    if revision != DATASET_REVISION:
        raise RuntimeError(
            "TinyStories changed upstream. Review the new revision before using it: "
            f"expected {DATASET_REVISION}, found {revision}."
        )

    spec = PROFILE_SPECS[profile]
    offsets = deterministic_batch_offsets(spec["stories"], spec["offset_seed"])
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET_REPO,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        payload = _read_json_url(f"{DATASET_API}?{query}")
        if payload.get("partial"):
            raise RuntimeError(f"Dataset API returned a partial page at offset {offset}")
        for row in payload["rows"]:
            if row.get("truncated_cells"):
                raise RuntimeError(f"Dataset API truncated row {row.get('row_idx')}")
            rows.append({"row_idx": int(row["row_idx"]), "text": str(row["row"]["text"])})
        time.sleep(0.20)
    rows = rows[: spec["stories"]]
    raw_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    raw_path.write_text(raw_text)
    manifest = {
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "source_split": "train",
        "selection": "deterministic non-overlapping 100-row pages, then truncate",
        "profile": profile,
        "requested_stories": spec["stories"],
        "downloaded_stories": len(rows),
        "offset_seed": spec["offset_seed"],
        "page_offsets": offsets,
        "raw_sha256": sha256_bytes(raw_text.encode()),
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api": DATASET_API,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def detokenize(tokens: Sequence[str]) -> str:
    """Readable reconstruction; tokenizing it again preserves the token sequence."""
    text = " ".join(token for token in tokens if token not in SPECIAL_TOKENS)
    text = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r'([\"“])\s+', r"\1", text)
    text = re.sub(r"\s+([\"”])", r"\1", text)
    text = re.sub(r"\s+'\s*", "'", text)
    return text.strip()


def stable_story_split(text: str, seed: int = 31415) -> str:
    digest = hashlib.sha256(f"{seed}\0{normalize_text(text)}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "validation"
    return "test"


@dataclass
class Story:
    row_idx: int
    doc_id: str
    text: str
    tokens: list[str]
    split: str


@dataclass
class Vocabulary:
    itos: list[str]
    stoi: dict[str, int]
    counts: dict[str, int]
    max_vocab: int
    min_freq: int

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode_tokens(self, tokens: Sequence[str], boundaries: bool = True) -> list[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in tokens]
        return ([self.bos_id] + ids + [self.eos_id]) if boundaries else ids

    def decode_ids(self, ids: Sequence[int], skip_special: bool = True) -> list[str]:
        tokens = [self.itos[int(idx)] for idx in ids]
        return [token for token in tokens if token not in SPECIAL_TOKENS] if skip_special else tokens


@dataclass
class Corpus:
    stories: dict[str, list[Story]]
    vocab: Vocabulary
    manifest: dict[str, Any]
    audit: dict[str, Any]


def _load_raw_stories(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_corpus(
    data_root: Path | str,
    profile: str = "smoke",
    *,
    download: bool = True,
    split_seed: int = 31415,
) -> Corpus:
    if profile not in PROFILE_SPECS:
        raise ValueError(profile)
    profile_dir = Path(data_root) / profile
    if download:
        manifest = fetch_bounded_tinystories(profile_dir, profile)
    else:
        manifest = json.loads((profile_dir / "source_manifest.json").read_text())
    raw_rows = _load_raw_stories(profile_dir / "stories.jsonl")

    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in raw_rows:
        normalized = normalize_text(row["text"])
        doc_id = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        if doc_id in unique:
            if normalize_text(unique[doc_id]["text"]) != normalized:
                raise RuntimeError("Unexpected SHA-256 prefix collision")
            duplicate_count += 1
            continue
        unique[doc_id] = row

    stories = {"train": [], "validation": [], "test": []}
    for doc_id, row in unique.items():
        tokens = tokenize(row["text"])
        split = stable_story_split(row["text"], split_seed)
        stories[split].append(
            Story(int(row["row_idx"]), doc_id, str(row["text"]), tokens, split)
        )
    for split in stories:
        stories[split].sort(key=lambda story: story.doc_id)

    spec = PROFILE_SPECS[profile]
    counts = Counter(token for story in stories["train"] for token in story.tokens)
    lexical = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= spec["min_freq"]
    ][: spec["max_vocab"] - len(SPECIAL_TOKENS)]
    itos = list(SPECIAL_TOKENS) + lexical
    vocab = Vocabulary(
        itos=itos,
        stoi={token: idx for idx, token in enumerate(itos)},
        counts=dict(counts),
        max_vocab=spec["max_vocab"],
        min_freq=spec["min_freq"],
    )

    split_ids = {split: {story.doc_id for story in split_stories} for split, split_stories in stories.items()}
    overlap = {
        "train_validation": len(split_ids["train"] & split_ids["validation"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "validation_test": len(split_ids["validation"] & split_ids["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Document leakage detected: {overlap}")

    oov: dict[str, dict[str, float | int]] = {}
    for split, split_stories in stories.items():
        tokens = [token for story in split_stories for token in story.tokens]
        unknown = sum(token not in vocab.stoi for token in tokens)
        oov[split] = {
            "tokens": len(tokens),
            "unknown": unknown,
            "rate": unknown / max(1, len(tokens)),
        }
    audit = {
        "split_seed": split_seed,
        "documents": {split: len(value) for split, value in stories.items()},
        "duplicates_removed_before_split": duplicate_count,
        "document_overlap": overlap,
        "vocab_size": len(vocab.itos),
        "vocab_fit_on": "train only",
        "oov": oov,
        "story_length_tokens": {
            split: {
                "min": min((len(story.tokens) for story in value), default=0),
                "median": float(np.median([len(story.tokens) for story in value])) if value else 0.0,
                "max": max((len(story.tokens) for story in value), default=0),
            }
            for split, value in stories.items()
        },
    }
    return Corpus(stories, vocab, manifest, audit)


def make_windows(
    stories: Sequence[Story], vocab: Vocabulary, context_len: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Create one target per window; contexts never cross story boundaries."""
    contexts: list[list[int]] = []
    targets: list[int] = []
    document_ids: list[str] = []
    for story in stories:
        ids = vocab.encode_tokens(story.tokens, boundaries=True)
        for target_pos in range(1, len(ids)):
            start = max(0, target_pos - context_len)
            context = ids[start:target_pos]
            context = [vocab.pad_id] * (context_len - len(context)) + context
            contexts.append(context)
            targets.append(ids[target_pos])
            document_ids.append(story.doc_id)
    return (
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        document_ids,
    )


def make_all_windows(corpus: Corpus, context_len: int) -> dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]]:
    return {
        split: make_windows(stories, corpus.vocab, context_len)
        for split, stories in corpus.stories.items()
    }


class FixedWindowMLP(nn.Module):
    """Learned word embeddings -> concatenate -> ReLU hidden layer -> logits."""

    def __init__(self, vocab_size: int, context_len: int, d_embed: int, hidden: int, pad_id: int = 0):
        super().__init__()
        self.context_len = context_len
        self.d_embed = d_embed
        self.hidden = hidden
        self.token_embedding = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.hidden_layer = nn.Linear(context_len * d_embed, hidden)
        self.vocab_head = nn.Linear(hidden, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(context_ids)
        concatenated = embedded.reshape(context_ids.shape[0], self.context_len * self.d_embed)
        hidden = F.relu(self.hidden_layer(concatenated))
        return self.vocab_head(hidden)


class CausalAttentionLM(nn.Module):
    """One causal attention head, output projection, residual, then prediction MLP."""

    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        d_k: int,
        d_v: int,
        hidden: int,
        pad_id: int = 0,
    ):
        super().__init__()
        self.context_len = context_len
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.hidden = hidden
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(context_len, d_model)
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.W_O = nn.Linear(d_v, d_model, bias=False)
        self.hidden_layer = nn.Linear(d_model, hidden)
        self.vocab_head = nn.Linear(hidden, vocab_size)

    def forward_details(self, context_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, length = context_ids.shape
        positions = torch.arange(length, device=context_ids.device)
        X = self.token_embedding(context_ids) + self.position_embedding(positions)[None, :, :]
        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        future = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=context_ids.device), diagonal=1
        )
        # Left padding represents absent history, not a token the prediction
        # query should spend probability mass on.  Padded query rows retain
        # their causal prefix so every softmax row stays finite; real query
        # rows mask padded keys as well as future keys.
        real_tokens = context_ids.ne(self.pad_id)
        padded_key_for_real_query = (~real_tokens[:, None, :]) & real_tokens[:, :, None]
        attention_mask = future[None, :, :] | padded_key_for_real_query
        masked_scores = scores.masked_fill(attention_mask, float("-inf"))
        weights = F.softmax(masked_scores, dim=-1)
        messages = weights @ V
        updates = self.W_O(messages)
        contextual = X + updates
        hidden = F.relu(self.hidden_layer(contextual))
        logits_all = self.vocab_head(hidden)
        return {
            "X": X,
            "Q": Q,
            "K": K,
            "V": V,
            "scores": scores,
            "masked_scores": masked_scores,
            "attention_mask": attention_mask,
            "weights": weights,
            "messages": messages,
            "updates": updates,
            "contextual": contextual,
            "hidden": hidden,
            "logits_all": logits_all,
            "logits": logits_all[:, -1, :],
        }

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        # Training supervises one next token per window, so only the final
        # causal query contributes to the loss.  Computing that row directly
        # is exactly equivalent to taking the final row from forward_details,
        # while avoiding unused logits for every earlier position.
        _, length = context_ids.shape
        positions = torch.arange(length, device=context_ids.device)
        X = self.token_embedding(context_ids) + self.position_embedding(positions)[None, :, :]
        q_last = self.W_Q(X[:, -1:, :])
        K, V = self.W_K(X), self.W_V(X)
        scores_last = q_last @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        padded_keys = context_ids.eq(self.pad_id)[:, None, :]
        weights_last = F.softmax(scores_last.masked_fill(padded_keys, float("-inf")), dim=-1)
        message_last = weights_last @ V
        contextual_last = X[:, -1, :] + self.W_O(message_last).squeeze(1)
        hidden_last = F.relu(self.hidden_layer(contextual_last))
        return self.vocab_head(hidden_last)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def approximate_multiplies(model: nn.Module) -> int:
    """Pedagogical forward-pass multiply count for one example (not measured FLOPs)."""
    vocab_size = model.vocab_head.out_features
    if hasattr(model, "context_len") and hasattr(model, "d_embed") and not hasattr(model, "W_Q"):
        hidden = model.hidden_layer.out_features
        return (
            model.context_len * model.d_embed * hidden
            + hidden * vocab_size
        )
    if hasattr(model, "W_Q") and hasattr(model, "W_K") and hasattr(model, "W_V"):
        T = model.context_len
        d_model = model.token_embedding.embedding_dim
        d_k = model.W_Q.out_features
        d_v = model.W_V.out_features
        hidden = model.hidden_layer.out_features
        return (
            T * d_model * (2 * d_k + d_v)
            + T * T * d_k
            + T * T * d_v
            + T * d_v * d_model
            + d_model * hidden
            + hidden * vocab_size
        )
    raise TypeError(type(model))


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 400
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 11
    eval_every: int = 100
    eval_examples: int = 20_000
    gradient_clip: float = 1.0


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device | str,
    batch_size: int = 1024,
    max_examples: int | None = None,
) -> float:
    model.eval()
    if max_examples is not None:
        X, y = X[:max_examples], y[:max_examples]
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(y), batch_size):
        xb = X[start : start + batch_size].to(device, non_blocking=True)
        yb = y[start : start + batch_size].to(device, non_blocking=True)
        total_loss += float(F.cross_entropy(model(xb), yb, reduction="sum").cpu())
        total_tokens += len(yb)
    return total_loss / max(1, total_tokens)


def train_model(
    model: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    validation_X: torch.Tensor,
    validation_y: torch.Tensor,
    config: TrainConfig,
    *,
    device: torch.device | str = "auto",
    capture_embeddings: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    seed_everything(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 10_000)
    trace: list[dict[str, float | int]] = []
    embedding_trace: dict[str, np.ndarray] = {}
    if capture_embeddings:
        embedding_trace["step_0"] = model.token_embedding.weight.detach().cpu().numpy().copy()
    start_time = time.perf_counter()
    midpoint = max(1, config.steps // 2)
    last_loss = float("nan")
    best_validation_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, config.steps + 1):
        indices = torch.randint(0, len(train_y), (config.batch_size,), generator=generator)
        xb = train_X[indices].to(device, non_blocking=True)
        yb = train_y[indices].to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()
        finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not finite:
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        if config.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if capture_embeddings and step == midpoint:
            embedding_trace[f"step_{step}"] = model.token_embedding.weight.detach().cpu().numpy().copy()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            validation_loss = evaluate_loss(
                model,
                validation_X,
                validation_y,
                device=device,
                max_examples=config.eval_examples,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            trace.append(
                {
                    "step": step,
                    "tokens_seen": step * config.batch_size,
                    "train_batch_loss": last_loss,
                    "validation_loss": validation_loss,
                }
            )
    if best_state is None:
        raise RuntimeError("Training finished without a validation checkpoint")
    model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - start_time
    if capture_embeddings:
        for name in list(embedding_trace):
            if name.startswith("step_") and int(name.removeprefix("step_")) > best_step:
                del embedding_trace[name]
        embedding_trace[f"selected_step_{best_step}"] = (
            model.token_embedding.weight.detach().cpu().numpy().copy()
        )
    result: dict[str, Any] = {
        "train_config": asdict(config),
        "parameter_count": count_parameters(model),
        "approximate_forward_multiplies": approximate_multiplies(model),
        "tokens_seen": config.steps * config.batch_size,
        "selected_step": best_step,
        "selected_tokens_seen": best_step * config.batch_size,
        "checkpoint_selection": "lowest validation loss on the fixed selection slice",
        "selection_slice_validation_loss": best_validation_loss,
        "runtime_seconds": runtime,
        "final_train_batch_loss": last_loss,
        "validation_loss": evaluate_loss(model, validation_X, validation_y, device=device),
        "trace": trace,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    result["validation_perplexity"] = math.exp(result["validation_loss"])
    return result, embedding_trace


def tiny_batch_overfit(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    steps: int = 160,
    learning_rate: float = 2e-2,
) -> tuple[float, float]:
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    model.to(device)
    xb, yb = X[:16].to(device), y[:16].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial = float(F.cross_entropy(model(xb), yb).cpu())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(F.cross_entropy(model(xb), yb).cpu())
    return initial, final


@torch.no_grad()
def generate_text(
    model: nn.Module,
    vocab: Vocabulary,
    prompt: str,
    *,
    context_len: int,
    max_new_tokens: int = 60,
    temperature: float = 0.9,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    model.eval().to(device)
    prompt_tokens = tokenize(prompt)
    generated = vocab.encode_tokens(prompt_tokens, boundaries=False)
    history = [vocab.bos_id] + generated
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max_new_tokens):
        context = history[-context_len:]
        context = [vocab.pad_id] * (context_len - len(context)) + context
        logits = model(torch.tensor([context], dtype=torch.long, device=device))[0].cpu()
        logits[[vocab.pad_id, vocab.bos_id, vocab.unk_id]] = float("-inf")
        probabilities = F.softmax(logits / temperature, dim=-1)
        next_id = int(torch.multinomial(probabilities, 1, generator=generator))
        if next_id == vocab.eos_id:
            break
        history.append(next_id)
        generated.append(next_id)
    return detokenize(vocab.decode_ids(generated, skip_special=True))


def model_state_to_npz(
    path: Path | str,
    model: nn.Module,
    metadata: dict[str, Any],
    embedding_trace: dict[str, np.ndarray] | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        f"state__{name}": tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    if embedding_trace:
        for name, array in embedding_trace.items():
            arrays[f"embedding__{name}"] = np.asarray(array)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_model_npz(path: Path | str, model: nn.Module) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        state = {
            name.removeprefix("state__"): torch.from_numpy(data[name].copy())
            for name in data.files
            if name.startswith("state__")
        }
        model.load_state_dict(state)
        metadata = json.loads(str(data["metadata_json"]))
        embedding_trace = {
            name.removeprefix("embedding__"): data[name].copy()
            for name in data.files
            if name.startswith("embedding__")
        }
    return metadata, embedding_trace


def cosine_neighbors(
    embedding: np.ndarray,
    vocab: Vocabulary,
    token: str,
    top_k: int = 8,
) -> list[tuple[str, float]]:
    idx = vocab.stoi[token]
    matrix = embedding.astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    similarities = matrix @ matrix[idx]
    similarities[list(range(len(SPECIAL_TOKENS)))] = -np.inf
    similarities[idx] = -np.inf
    best = np.argsort(-similarities)[:top_k]
    return [(vocab.itos[int(other)], float(similarities[other])) for other in best]


def fit_common_pca(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fit one 2-D basis across checkpoints so plots share an orientation."""
    stacked = np.concatenate([np.asarray(array, dtype=np.float64) for array in arrays], axis=0)
    mean = stacked.mean(axis=0)
    _, _, vt = np.linalg.svd(stacked - mean, full_matrices=False)
    return mean, vt[:2].T


def project_with_basis(array: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(array, dtype=np.float64) - mean) @ basis


def top_probability_changes(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    vocab: Vocabulary,
    top_k: int = 8,
) -> list[dict[str, float | str]]:
    before = F.softmax(logits_before.detach().cpu(), dim=-1)
    after = F.softmax(logits_after.detach().cpu(), dim=-1)
    delta = after - before
    indices = torch.argsort(delta.abs(), descending=True)[:top_k]
    return [
        {
            "token": vocab.itos[int(idx)],
            "before": float(before[idx]),
            "after": float(after[idx]),
            "delta": float(delta[idx]),
        }
        for idx in indices
    ]


def attention_invariants(model: CausalAttentionLM, context_ids: torch.Tensor) -> dict[str, float | bool]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(context_ids)
        weights = details["weights"]
        length = weights.shape[-1]
        future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        future_values = weights.detach().cpu()[:, future]
        row_error = float((weights.sum(dim=-1) - 1).abs().max().cpu())

        changed = context_ids.clone()
        pivot = max(1, length // 2)
        changed[:, pivot:] = torch.flip(changed[:, pivot:], dims=[1])
        original = model.forward_details(context_ids)["contextual"]
        altered = model.forward_details(changed)["contextual"]
        prefix_error = float((original[:, :pivot] - altered[:, :pivot]).abs().max().cpu())
    return {
        "max_future_weight": float(future_values.abs().max()) if future_values.numel() else 0.0,
        "max_row_sum_error": row_error,
        "max_earlier_representation_change_after_future_edit": prefix_error,
        "all_finite": bool(torch.isfinite(details["logits"]).all()),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def environment_summary() -> dict[str, Any]:
    if torch.cuda.is_available():
        accelerator = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        accelerator = "Apple Metal (MPS)"
    else:
        accelerator = "cpu"
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "accelerator": accelerator,
        "platform": os.uname().sysname + " " + os.uname().machine,
    }
