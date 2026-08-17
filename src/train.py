"""Stage 4 - TRAIN.

Trains the candidate-scoring policy on the elite trajectories produced by
`generate_episodes.py`. The objective is imitation of the *good* games only
(cross-entropy / self-imitation): each decision is a classification over that
step's legal actions, weighted by how strong the episode it came from was.

    python src/train.py --tag bootstrap --epochs 12

Outputs:
    models/<name>.keras          trained policy+value network
    models/<name>_meta.json      dims, normalization, training config, metrics
    reports/training_<name>.md   curves + held-out metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data" / "episodes"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def load_shards(tags: list[str]) -> dict[str, np.ndarray]:
    files: list[Path] = []
    for tag in tags:
        files.extend(sorted(EPISODES.glob(f"{tag}_shard*.npz")))
    if not files:
        raise SystemExit(f"no shards found for tags {tags} - run src/generate_episodes.py first")
    parts: dict[str, list[np.ndarray]] = {}
    for f in files:
        with np.load(f) as z:
            for key in z.files:
                parts.setdefault(key, []).append(z[key])
        print(f"[train] loaded {f.name}")
    return {k: np.concatenate(v) for k, v in parts.items()}


def split_by_episode(data: dict[str, np.ndarray], val_frac: float, seed: int = 0):
    """Hold out whole games, never individual decisions - decisions inside one
    game are heavily correlated and would leak across the split."""
    ep_ids = np.unique(data["episode_ids"])
    rng = np.random.default_rng(seed)
    rng.shuffle(ep_ids)
    n_val = max(int(len(ep_ids) * val_frac), 1)
    val_ids = set(ep_ids[:n_val].tolist())
    is_val = np.array([e in val_ids for e in data["episode_ids"]])
    return ~is_val, is_val


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Civ 6 policy/value network.")
    ap.add_argument("--tag", nargs="+", default=["bootstrap"], help="episode tag(s) to train on")
    ap.add_argument("--name", default=None, help="model name (default: policy_<first tag>)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--value-weight", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None, help="warm-start from an existing .keras model")
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    from model import build_model

    tf.keras.utils.set_random_seed(args.seed)
    name = args.name or f"policy_{args.tag[0]}"
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    data = load_shards(args.tag)
    # candidates dominate memory (N x C x A); keep them in float16 and cast per
    # batch inside the input pipeline instead of materializing a float32 copy.
    states = data["states"].astype(np.float32)
    cands = data["candidates"]
    masks = data["masks"].astype(np.float32)
    choices = data["choices"].astype(np.int32)
    returns = data["returns"].astype(np.float32)
    print(f"[train] {len(states)} decisions | state {states.shape[1]} | "
          f"candidates {cands.shape[1]}x{cands.shape[2]} ({cands.nbytes / 1e6:.0f} MB fp16) | "
          f"{len(np.unique(data['episode_ids']))} games")

    # Normalize the value target to the observed score range; keep the constants
    # in the metadata so evaluation can invert it.
    ret_mean, ret_std = float(returns.mean()), float(returns.std() + 1e-6)
    value_target = (returns - ret_mean) / ret_std

    # Sample weights: decisions from better games matter more, and later turns
    # (which are closer to the payoff) matter slightly more than opening moves.
    rank = (returns - returns.min()) / (returns.max() - returns.min() + 1e-6)
    weights = (0.5 + rank).astype(np.float32)

    train_ix, val_ix = split_by_episode(data, args.val_frac, seed=args.seed)
    print(f"[train] split: {train_ix.sum()} train / {val_ix.sum()} val decisions")

    model = build_model(states.shape[1], cands.shape[2], hidden=args.hidden,
                        embed=args.embed, dropout=args.dropout)
    if args.init_from:
        # Same architecture, so copying the weights across is enough and keeps
        # the fresh optimizer state (each round trains on a new elite set).
        previous = keras.models.load_model(args.init_from, compile=False)
        model.set_weights(previous.get_weights())
        print(f"[train] warm-started from {args.init_from}")
    model.summary(print_fn=lambda s: print("    " + s))

    optimizer = keras.optimizers.Adam(args.lr)

    def make_ds(ix: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((
            {"state": states[ix], "actions": cands[ix], "mask": masks[ix]},
            {"choice": choices[ix], "value": value_target[ix], "weight": weights[ix]},
        ))
        if training:
            ds = ds.shuffle(min(50_000, int(ix.sum())), seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.batch(args.batch_size)
        ds = ds.map(lambda x, y: ({**x, "actions": tf.cast(x["actions"], tf.float32)}, y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = make_ds(train_ix, True), make_ds(val_ix, False)

    @tf.function
    def step(inputs, targets, training: bool):
        with tf.GradientTape() as tape:
            out = model(inputs, training=training)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets["choice"], logits=out["logits"])
            policy_loss = tf.reduce_mean(ce * targets["weight"])
            value_loss = tf.reduce_mean(tf.square(out["value"] - targets["value"]))
            loss = policy_loss + args.value_weight * value_loss + tf.add_n(model.losses or [0.0])
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        top1 = tf.reduce_mean(tf.cast(tf.equal(
            tf.cast(tf.argmax(out["logits"], axis=1), tf.int32), targets["choice"]), tf.float32))
        top3 = tf.reduce_mean(tf.cast(tf.math.in_top_k(targets["choice"], out["logits"], 3), tf.float32))
        return loss, policy_loss, value_loss, top1, top3

    history: list[dict] = []
    best_val = float("inf")
    best_path = MODELS / f"{name}.keras"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        agg = np.zeros(5)
        n = 0
        for inputs, targets in train_ds:
            vals = step(inputs, targets, True)
            agg += np.array([float(v) for v in vals])
            n += 1
        tr = agg / max(n, 1)

        vagg = np.zeros(5)
        m = 0
        for inputs, targets in val_ds:
            vals = step(inputs, targets, False)
            vagg += np.array([float(v) for v in vals])
            m += 1
        va = vagg / max(m, 1)

        row = {"epoch": epoch, "train_loss": tr[0], "train_policy": tr[1], "train_value": tr[2],
               "train_top1": tr[3], "train_top3": tr[4], "val_loss": va[0], "val_policy": va[1],
               "val_value": va[2], "val_top1": va[3], "val_top3": va[4], "secs": time.time() - t0}
        history.append(row)
        flag = ""
        if va[1] < best_val:
            best_val = va[1]
            model.save(best_path)
            flag = "  <- saved"
        print(f"[train] epoch {epoch:2d}/{args.epochs}  loss {tr[0]:.4f}/{va[0]:.4f}  "
              f"policy {tr[1]:.4f}/{va[1]:.4f}  top1 {tr[3]:.3f}/{va[3]:.3f}  "
              f"top3 {tr[4]:.3f}/{va[4]:.3f}  value {va[2]:.3f}  {row['secs']:.0f}s{flag}")

    meta = {
        "name": name, "tags": args.tag, "state_dim": int(states.shape[1]),
        "action_dim": int(cands.shape[2]), "max_candidates": int(cands.shape[1]),
        "n_decisions": int(len(states)), "n_games": int(len(np.unique(data["episode_ids"]))),
        "return_mean": ret_mean, "return_std": ret_std,
        "elite_score_mean": float(returns.mean()), "elite_score_max": float(returns.max()),
        "config": vars(args), "history": history,
        "best_val_policy_loss": best_val,
    }
    (MODELS / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    last = history[-1]
    lines = [
        f"# Training - `{name}`", "",
        f"- data: {meta['n_decisions']} decisions from {meta['n_games']} elite games "
        f"(tags: {', '.join(args.tag)})",
        f"- elite score mean **{meta['elite_score_mean']:.1f}**, max {meta['elite_score_max']:.1f}",
        f"- network: state {meta['state_dim']} -> {args.hidden}/{args.embed}, "
        f"candidate scorer over {meta['max_candidates']} slots, {model.count_params():,} params",
        f"- best validation policy loss: **{best_val:.4f}**",
        f"- final val top-1 **{last['val_top1']:.3f}**, top-3 **{last['val_top3']:.3f}**",
        "", "| epoch | train loss | val loss | val top-1 | val top-3 | val value MSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | "
              f"{h['val_top1']:.3f} | {h['val_top3']:.3f} | {h['val_value']:.3f} |" for h in history]
    (REPORTS / f"training_{name}.md").write_text("\n".join(lines) + "\n")
    print(f"[train] saved {best_path.relative_to(ROOT)} and reports/training_{name}.md")


if __name__ == "__main__":
    main()
