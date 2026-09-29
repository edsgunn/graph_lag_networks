#!/usr/bin/env python3
"""Graph Lag Networks: a word-level chain LM trained with evosax Open_ES.

Python 3.10+; CPU works on Apple Silicon. Suggested isolated installation:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install 'jax[cpu]==0.5.3' 'flax==0.10.6' 'evosax==0.2.0' 'optax==0.2.5' \
      'numpy==2.2.6' 'orbax-checkpoint==0.11.13'

Train on Alice's Adventures in Wonderland (downloaded once):
  python graph_lag_lm.py
Train on your own UTF-8 text:
  python graph_lag_lm.py --text corpus.txt --nodes 8 --seq-len 256
Quick run:
  python graph_lag_lm.py --generations 20 --population 16 --hidden 8 \
      --vocab-size 256 --seq-len 32 --warmup 8 --batch-size 2 --eval-every 10
Check causal propagation without downloading data:
  python graph_lag_lm.py --self-test
Generate from a saved model:
  python graph_lag_lm.py --checkpoint gln_run/best.npz --prompt 'alice was'

Architecture: h_i[t] = GRU(h_i[t-1], h_{i-1}[t-1], h_{i+1}[t-1], input_i[t]).
Only input_0 receives the current word. All nodes update simultaneously, once
per token. Logits come from the NEW h_0[t] and predict word[t+1]. Open ends;
no wraparound, sweeps, global pooling, attention, or extra communication rounds.
The shared GRU permits depth specialization through state/context. Its weights
are identical at all nodes. Boundary flags only identify the chain endpoints.

Fitness is ordinary mean teacher-forced next-token cross entropy (natural-log
units), with no auxiliary objectives, regularization or rank transformation.
Open_ES uses antithetic parameter perturbations and Adam on its ES estimate.
Every candidate uses the same word windows, with fixed weights for the entire
warmup + scored rollout. Each window starts at zero state. Warmup is unscored
context; this does NOT implement persistent ES or credit across generations.

The vocabulary is fit on the first 90% of tokens; the last 10% is validation.
Words are lowercased and punctuation is tokenized separately. Rare/unseen words
map to <unk>, so perplexity is over this restricted vocabulary (OOV rates logged).
Random training windows never cross the split. Validation uses fixed windows
with disjoint scored spans, and selects the best distribution-mean checkpoint.
Language CE improvement alone does not demonstrate use of distant nodes.

Outputs: config.json, vocab.json, metrics.csv, best.npz, last.npz, sample.txt,
and learning_curve.png if matplotlib is available. NPZ files are inference
checkpoints, not resumable optimizer checkpoints. Existing runs are protected.
Population chunking bounds activation memory, but Open_ES still stores O(P*D)
perturbed parameters. Increase sequence length to expose longer dependencies;
increase --population when the ES estimate is noisy. First JIT calls are slower.

References / APIs:
  https://github.com/RobertTLange/evosax (v0.2.0)
  https://flax.readthedocs.io/ ; https://docs.jax.dev/
Corpus: Lewis Carroll, https://www.gutenberg.org/ebooks/11 (public-domain text).
"""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
import urllib.request
import warnings

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from flax import linen as nn
import optax
from evosax.algorithms import Open_ES
from evosax.core.fitness_shaping import identity_fitness_shaping_fn


class ChainLM(nn.Module):
    vocab_size: int
    nodes: int
    hidden: int

    @nn.compact
    def __call__(self, old, token):
        """old: [batch,nodes,hidden], token: [batch]; one synchronous tick."""
        embedding = nn.Embed(self.vocab_size, self.hidden, name="embedding")
        word = embedding(token)
        # Both messages read OLD state. Never replace state in a node-wise loop.
        zero = jnp.zeros_like(old[:, :1])
        left = jnp.concatenate((zero, old[:, :-1]), axis=1)
        right = jnp.concatenate((old[:, 1:], zero), axis=1)
        external = jnp.zeros_like(old).at[:, 0, :].set(word)
        positions = jnp.arange(self.nodes)
        flags = jnp.stack((positions == 0, positions == self.nodes - 1), -1)
        flags = jnp.broadcast_to(flags, (*old.shape[:2], 2)).astype(old.dtype)
        local = jnp.concatenate((left, right, external, flags), axis=-1)
        new, _ = nn.GRUCell(features=self.hidden, name="shared_cell")(old, local)
        # Weight tying keeps the vocabulary cost manageable for ES.
        bias = self.param("output_bias", nn.initializers.zeros, (self.vocab_size,))
        logits = embedding.attend(new[:, 0, :]) + bias
        return new, logits


def tokenize(text):
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*|[^\w\s]", text, flags=re.UNICODE)


def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG", text)
    return text[:end.start()] if end else text


def read_corpus(args):
    if args.text:
        return Path(args.text).read_text(encoding="utf-8"), str(Path(args.text).resolve())
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "alice.txt"
    url = "https://www.gutenberg.org/ebooks/11.txt.utf-8"
    if not path.exists():
        print("Downloading Alice's Adventures in Wonderland...", flush=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GraphLagLM/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8-sig")
            if "alice" not in text.lower() or len(text) < 10000:
                raise ValueError("Downloaded response does not look like the corpus.")
            path.write_text(text, encoding="utf-8")
        except Exception as exc:
            raise RuntimeError("Corpus download failed. Supply a local UTF-8 file with --text.") from exc
    return path.read_text(encoding="utf-8"), url


def prepare_data(args):
    text, source = read_corpus(args)
    words = tokenize(strip_gutenberg(text))
    split = int(len(words) * 0.9)
    train_words, val_words = words[:split], words[split:]
    counts = Counter(train_words)
    vocab = ["<unk>"] + [w for w, _ in counts.most_common(args.vocab_size - 1)]
    lookup = {w: i for i, w in enumerate(vocab)}
    train = np.array([lookup.get(w, 0) for w in train_words], dtype=np.int32)
    val = np.array([lookup.get(w, 0) for w in val_words], dtype=np.int32)
    width = args.warmup + args.seq_len + 1
    if min(len(train), len(val)) < width:
        raise ValueError(f"Both splits need at least {width} tokens; got {len(train)}, {len(val)}.")
    # Each validation window has preceding warmup; scored spans do not overlap.
    starts = np.arange(0, len(val) - width + 1, args.seq_len)
    if len(starts) > args.val_windows:
        starts = starts[np.linspace(0, len(starts) - 1, args.val_windows, dtype=int)]
    validation = val[starts[:, None] + np.arange(width)]
    rng = np.random.default_rng(args.seed + 19)
    starts_train = rng.integers(0, len(train) - width + 1, args.batch_size)
    monitor = train[starts_train[:, None] + np.arange(width)]
    stats = dict(source=source, sha256=hashlib.sha256(text.encode()).hexdigest(),
                 train_tokens=len(train), validation_tokens=len(val), vocabulary_size=len(vocab),
                 train_unk_fraction=float(np.mean(train == 0)),
                 validation_unk_fraction=float(np.mean(val == 0)),
                 validation_windows=len(validation), validation_scored_tokens=len(validation)*args.seq_len)
    return train, validation, monitor, vocab, stats


def initialize(model, key):
    state = jnp.zeros((1, model.nodes, model.hidden), dtype=jnp.float32)
    params = model.init(key, state, jnp.zeros((1,), jnp.int32))["params"]
    flat, unravel = ravel_pytree(params)
    return flat, unravel


def make_loss(model, unravel, warmup):
    def loss(flat, windows):
        params = unravel(flat)
        h = jnp.zeros((windows.shape[0], model.nodes, model.hidden), jnp.float32)
        x, y = windows[:, :-1].T, windows[:, 1:].T

        def tick(carry, item):
            h, total = carry
            token, target, t = item
            h, logits = model.apply({"params": params}, h, token)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, target).sum()
            total = total + jnp.where(t >= warmup, ce, 0.0)
            return (h, total), None  # No activation/logit history is accumulated.

        (_, total), _ = jax.lax.scan(tick, (h, jnp.float32(0)), (x, y, jnp.arange(x.shape[0])))
        return total / (windows.shape[0] * (x.shape[0] - warmup))
    return loss


def save_checkpoint(path, flat, args, vocab, generation, validation_ce):
    metadata = {"nodes": args.nodes, "hidden": args.hidden, "vocab_size": len(vocab),
                "generation": generation, "validation_ce": validation_ce,
                "tokenizer": "lowercase_unicode_words_and_punctuation_v1"}
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, flat=np.asarray(flat), vocab=np.array(vocab),
                            metadata=np.array(json.dumps(metadata)))
    tmp.replace(path)


def generate(model, unravel, flat, vocab, prompt, count, temperature, seed):
    lookup = {w: i for i, w in enumerate(vocab)}
    tokens = tokenize(prompt)
    if not tokens:
        raise ValueError("The prompt must contain at least one token.")
    params = unravel(jnp.asarray(flat))
    step = jax.jit(lambda h, x: model.apply({"params": params}, h, x))
    h = jnp.zeros((1, model.nodes, model.hidden), jnp.float32)
    for word in tokens:
        h, logits = step(h, jnp.array([lookup.get(word, 0)], jnp.int32))
    key = jax.random.key(seed)
    result = list(tokens)
    for _ in range(count):
        key, sample_key = jax.random.split(key)
        # Suppress unknown tokens only when displaying samples, never for loss.
        scores = logits[0].at[0].set(-jnp.inf)
        nxt = int(jnp.argmax(scores) if temperature == 0 else
                  jax.random.categorical(sample_key, scores / temperature))
        result.append(vocab[nxt])
        h, logits = step(h, jnp.array([nxt], jnp.int32))
    text = " ".join(result)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


def self_test():
    """Check both causal cones, no wraparound, and the teacher-forcing shift."""
    model = ChainLM(vocab_size=11, nodes=5, hidden=8)
    flat, unravel = initialize(model, jax.random.key(7))
    params = unravel(flat)
    step = jax.jit(lambda h, x: model.apply({"params": params}, h, x))
    a = b = jnp.zeros((1, 5, 8))
    for t in range(5):
        a, la = step(a, jnp.array([1 if t == 0 else 3]))
        b, lb = step(b, jnp.array([2 if t == 0 else 3]))
        np.testing.assert_array_equal(np.asarray(a[:, t+1:]), np.asarray(b[:, t+1:]))
        assert np.max(np.abs(np.asarray(a[:, t] - b[:, t]))) > 1e-9
        if t == 0:
            assert not np.array_equal(np.asarray(la), np.asarray(lb)), "No immediate input response"
    for depth in range(1, 5):
        a = jnp.zeros((1, 5, 8))
        b = a.at[:, depth, :].set(0.5)
        for elapsed in range(1, depth + 1):
            a, la = step(a, jnp.array([3]))
            b, lb = step(b, jnp.array([3]))
            if elapsed < depth:
                np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))
            else:
                assert np.max(np.abs(np.asarray(la-lb))) > 1e-9
    window = jnp.array([[1, 4, 2, 5, 3, 6]], jnp.int32)
    loss = make_loss(model, unravel, warmup=2)
    h = jnp.zeros((1, 5, 8))
    manual = []
    for t in range(5):
        h, logits = step(h, window[:, t])
        if t >= 2:
            manual.append(float(optax.softmax_cross_entropy_with_integer_labels(logits, window[:, t+1])[0]))
    np.testing.assert_allclose(float(jax.jit(loss)(flat, window)), np.mean(manual), rtol=1e-6)
    one = ChainLM(vocab_size=11, nodes=1, hidden=8)
    f, u = initialize(one, jax.random.key(9))
    assert np.isfinite(float(jax.jit(make_loss(one, u, 0))(f, window)))
    print("PASS: one hop/tick in both directions, immediate readout, no wraparound, CE alignment, one-node case.")


def train(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory {out} is not empty; choose a new --out.")
    train_ids, validation, monitor, vocab, stats = prepare_data(args)
    out.mkdir(parents=True, exist_ok=True)
    model = ChainLM(len(vocab), args.nodes, args.hidden)
    key, init_key, es_key = jax.random.split(jax.random.key(args.seed), 3)
    flat, unravel = initialize(model, init_key)
    loss = make_loss(model, unravel, args.warmup)
    evaluate = jax.jit(loss)
    # Map candidates in chunks; all receive exactly the same windows.
    vector_loss = jax.vmap(loss, in_axes=(0, None))
    chunk = min(args.population_chunk, args.population)
    population_loss = jax.jit(lambda population, windows: jax.lax.map(
        lambda part: vector_loss(part, windows),
        population.reshape((-1, chunk, flat.size))).reshape((args.population,)))
    schedule = optax.exponential_decay(args.sigma, transition_steps=1,
                                      decay_rate=args.sigma_decay, end_value=args.sigma_min)
    strategy = Open_ES(population_size=args.population, solution=flat,
                       optimizer=optax.adam(args.learning_rate), std_schedule=schedule,
                       fitness_shaping_fn=identity_fitness_shaping_fn)
    es_params = strategy.default_params
    es_state = strategy.init(es_key, flat, es_params)
    versions = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "evosax", "optax", "numpy")}
    config = vars(args) | stats | {"parameters": int(flat.size), "versions": versions,
                                  "devices": [str(d) for d in jax.devices()]}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    validation, monitor = jnp.asarray(validation), jnp.asarray(monitor)
    counts = np.bincount(train_ids, minlength=len(vocab)).astype(float) + 1
    unigram = counts / counts.sum()
    target = np.asarray(validation)[:, args.warmup+1:]
    unigram_ce = float(-np.log(unigram[target]).mean())
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Devices: {jax.devices()} | parameters: {flat.size:,} | nodes: {args.nodes}", flush=True)
    print(f"Uniform CE: {math.log(len(vocab)):.4f}; training-unigram validation CE: {unigram_ce:.4f}", flush=True)
    if args.warmup + args.seq_len <= 2*(args.nodes-1):
        warnings.warn("Rollout too short to include a full input/deepest-node/output round trip.")
    rng = np.random.default_rng(args.seed)
    width = args.warmup + args.seq_len + 1
    best_ce, best_flat, best_generation = math.inf, flat, 0
    records = []
    start = time.perf_counter()
    fields = ["generation", "elapsed_s", "train_monitor_ce", "validation_ce", "validation_ppl",
              "population_mean_ce", "population_min_ce", "sigma", "unigram_validation_ce"]

    def record(generation, state, population_ce):
        nonlocal best_ce, best_flat, best_generation
        mean = strategy.get_mean(state)
        val_ce = float(evaluate(mean, validation))
        train_ce = float(evaluate(mean, monitor))
        if not np.isfinite([val_ce, train_ce]).all():
            raise FloatingPointError("Non-finite loss; reduce learning rate or sigma.")
        if val_ce < best_ce:
            best_ce, best_flat, best_generation = val_ce, mean, generation
            save_checkpoint(out / "best.npz", mean, args, vocab, generation, val_ce)
        save_checkpoint(out / "last.npz", mean, args, vocab, generation, val_ce)
        row = dict(generation=generation, elapsed_s=time.perf_counter()-start,
                   train_monitor_ce=train_ce, validation_ce=val_ce,
                   validation_ppl=math.exp(min(val_ce, 80)),
                   population_mean_ce=float(np.mean(population_ce)) if population_ce is not None else "",
                   population_min_ce=float(np.min(population_ce)) if population_ce is not None else "",
                   sigma=float(state.std), unigram_validation_ce=unigram_ce)
        records.append(row)
        writer.writerow(row)
        log.flush()
        print(f"gen {generation:5d} | train {train_ce:.4f} | val {val_ce:.4f} | "
              f"ppl {row['validation_ppl']:.2f} | {row['elapsed_s']:.1f}s", flush=True)

    with (out / "metrics.csv").open("w", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        print("Compiling validation and population rollouts...", flush=True)
        record(0, es_state, None)
        for generation in range(1, args.generations+1):
            starts = rng.integers(0, len(train_ids)-width+1, size=args.batch_size)
            windows = jnp.asarray(train_ids[starts[:, None] + np.arange(width)])
            key, ask_key, tell_key = jax.random.split(key, 3)
            population, es_state = strategy.ask(ask_key, es_state, es_params)
            fitness = population_loss(population, windows)
            if not bool(jnp.all(jnp.isfinite(fitness))):
                raise FloatingPointError("Non-finite population fitness.")
            es_state, _ = strategy.tell(tell_key, population, fitness, es_state, es_params)
            if generation % args.eval_every == 0 or generation == args.generations:
                record(generation, es_state, np.asarray(fitness))
    sample = generate(model, unravel, best_flat, vocab, args.prompt, args.sample_tokens,
                      args.temperature, args.seed+1)
    (out / "sample.txt").write_text(sample + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [r["generation"] for r in records]
        for name, label in [("train_monitor_ce", "Fixed training windows"), ("validation_ce", "Validation")]:
            ax.plot(xs, [r[name] for r in records], label=label)
        ax.axhline(unigram_ce, ls="--", color="gray", label="Training unigram (validation)")
        ax.set(xlabel="ES generation", ylabel="Next-token cross entropy (nats)", title="Graph Lag Network")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "learning_curve.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print(f"Best validation CE {best_ce:.4f} at generation {best_generation}; saved to {out}")
    print("Sample (not a quality guarantee):\n" + sample)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", help="UTF-8 corpus; default downloads Alice")
    p.add_argument("--cache", default=str(Path.home()/".cache"/"graph_lag_lm"))
    p.add_argument("--out", default="gln_run")
    p.add_argument("--nodes", type=int, default=6)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=128, help="Scored next-token predictions per window")
    p.add_argument("--warmup", type=int, default=32, help="Additional unscored context tokens")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--population-chunk", type=int, default=8)
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--sigma-min", type=float, default=0.005)
    p.add_argument("--sigma-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--val-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default="alice was")
    p.add_argument("--sample-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--checkpoint", help="Generate only from a saved .npz inference checkpoint")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("nodes", "hidden", "seq_len", "batch_size", "population", "population_chunk",
                 "generations", "eval_every", "val_windows"):
        if getattr(args, name) <= 0:
            p.error(name + " must be positive")
    if args.vocab_size < 2 or args.warmup < 0 or args.sample_tokens < 0 or args.temperature < 0:
        p.error("Invalid vocabulary, warmup, sample length, or temperature")
    if not np.isfinite([args.learning_rate, args.sigma, args.sigma_min, args.sigma_decay, args.temperature]).all():
        p.error("Floating-point arguments must be finite")
    if args.population % 2 or args.population % min(args.population, args.population_chunk):
        p.error("Population must be even and divisible by population chunk")
    if not (0 < args.sigma_min <= args.sigma and 0 < args.sigma_decay <= 1 and args.learning_rate > 0):
        p.error("Require 0 < sigma-min <= sigma, 0 < sigma-decay <= 1, learning-rate > 0")
    if args.checkpoint:
        with np.load(args.checkpoint, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            vocab, flat = data["vocab"].tolist(), data["flat"]
        model = ChainLM(meta["vocab_size"], meta["nodes"], meta["hidden"])
        template, unravel = initialize(model, jax.random.key(0))
        if template.shape != flat.shape:
            raise ValueError("Checkpoint parameter shape does not match its model metadata.")
        print(generate(model, unravel, flat, vocab, args.prompt, args.sample_tokens, args.temperature, args.seed))
    else:
        train(args)


if __name__ == "__main__":
    main()
