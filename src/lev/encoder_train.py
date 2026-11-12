"""Train and run the encoder decision head. Requires the `encoder` extra.

    uv run --extra encoder python -m lev.encoder_train train \\
        --data artifacts/en-joint-002/train.jsonl --out runs/enc-001/checkpoint \\
        --model answerdotai/ModernBERT-base --revision <sha> --max-len 3072

Inputs are the frozen Kev exports, so the encoder reads exactly the state and
question text Kev's `api_request` projection would send. Over-long inputs are
refused, never truncated: shortening a state silently changes the proposition
being judged, and `report` cannot tell that it happened.
"""

import argparse
import contextlib
import json
import random
import shutil
import time
from pathlib import Path

from .encoder import examples
from .io import encoded, jsonl, write


def parse():
    p = argparse.ArgumentParser(prog="lev.encoder_train")
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--revision", required=True)
    t.add_argument("--max-len", type=int, required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="cap on padded tokens per batch; bounds peak memory independently of --batch",
    )
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--warmup", type=float, default=0.06)
    t.add_argument(
        "--keep-epochs",
        type=int,
        default=0,
        help="also keep the last N per-epoch snapshots; 0 keeps only the final weights",
    )
    d = sub.add_parser("predict")
    d.add_argument("--data", required=True)
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-len", type=int, required=True)
    d.add_argument("--device", default="cuda")
    d.add_argument("--batch", type=int, default=32)
    d.add_argument("--max-tokens", type=int, default=8192)
    return p.parse_args()


def set_seed(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load(path):
    return examples(jsonl(path))


def tokenize(tokenizer, items, max_len):
    """Tokenize without truncation and refuse anything that does not fit."""
    encoded_rows = tokenizer(
        [x["text"] for x in items],
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    over = [
        (item["id"], item["question"], len(ids))
        for item, ids in zip(items, encoded_rows["input_ids"])
        if len(ids) > max_len
    ]
    if over:
        shown = ", ".join(f"{i}/{q}={n}" for i, q, n in over[:5])
        raise ValueError(
            f"{len(over)} of {len(items)} inputs exceed --max-len {max_len} "
            f"({shown}{', ...' if len(over) > 5 else ''}); raise --max-len rather than truncate"
        )
    return encoded_rows["input_ids"], encoded_rows["attention_mask"]


def dataset(items, ids, masks):
    import torch

    class Pairs(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            return {
                "input_ids": torch.tensor(ids[i], dtype=torch.long),
                "attention_mask": torch.tensor(masks[i], dtype=torch.long),
                "labels": torch.tensor(items[i]["label"], dtype=torch.long),
            }

    return Pairs()


def order(lengths, batch, budget, seed):
    """Length-bucketed batches: pad to the longest record in the batch, not the longest in the pack.

    Padding every short record up to the pack's worst case is what turns a 150M encoder
    into a 26 GiB allocation: SDPA's math kernel materialises a full L x L attention
    matrix, so a 3k-token pad costs ~36x the memory of a 500-token one. The frozen packs
    are mostly short, so bucketing removes nearly all of that waste.
    """
    rng = random.Random(seed)
    index = list(range(len(lengths)))
    rng.shuffle(index)
    index.sort(key=lambda i: lengths[i])
    groups, current, widest = [], [], 0
    for i in index:
        width = max(widest, lengths[i])
        if current and (len(current) >= batch or width * (len(current) + 1) > budget):
            groups.append(current)
            current, widest = [], 0
        current.append(i)
        widest = max(widest, lengths[i])
    if current:
        groups.append(current)
    return groups


def collate(tokenizer):
    import torch

    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("tokenizer has no pad token")

    def build(batch):
        width = max(len(x["input_ids"]) for x in batch)
        width += (-width) % 8
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item["input_ids"])
            ids[row, :n] = item["input_ids"]
            mask[row, :n] = item["attention_mask"]
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": torch.stack([x["labels"] for x in batch]),
        }

    return build


def context(dtype, device):
    """Autocast plus scaler. Weights stay fp32: bf16/fp16 are forward-only precision."""
    import torch

    if dtype == "fp32" or device.type != "cuda":
        return contextlib.nullcontext(), None
    if dtype == "fp16":
        return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")
    return torch.autocast("cuda", dtype=torch.bfloat16), None


def load_model(source, device, revision=None, fresh=False):
    from transformers import AutoModelForSequenceClassification

    if fresh:
        return AutoModelForSequenceClassification.from_pretrained(
            source, revision=revision, num_labels=2
        ).to(device)
    return AutoModelForSequenceClassification.from_pretrained(source).to(device)


def train(a):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    set_seed(a.seed)
    device = torch.device(a.device)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    items = load(a.data)
    ids, masks = tokenize(tokenizer, items, a.max_len)
    pairs = dataset(items, ids, masks)
    build = collate(tokenizer)
    groups = order([len(x) for x in ids], a.batch, a.max_tokens, a.seed)
    model = load_model(a.model, device, revision=a.revision, fresh=True)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        keep = parameter.ndim < 2 or "bias" in name or "norm" in name
        (no_decay if keep else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=a.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    steps = len(groups) // a.accum
    total = max(1, steps * a.epochs)
    warm = int(total * a.warmup)

    def rate(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return max(0.0, (total - step) / max(1, total - warm))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    autocast, scaler = context(a.dtype, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    history, started = [], time.time()
    print(
        f"train: {len(items)} pairs in {len(groups)} batches, {a.epochs} epochs, "
        f"batch {a.batch} / {a.max_tokens} padded tokens, max_len {a.max_len}, "
        f"dtype {a.dtype}, device {device}",
        flush=True,
    )
    for epoch in range(1, a.epochs + 1):
        random.Random(a.seed + epoch).shuffle(groups)
        loader = DataLoader(
            pairs, batch_sampler=groups, num_workers=0, collate_fn=build
        )
        model.train()
        running, seen, began = 0.0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            with autocast:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                loss = (
                    torch.nn.functional.cross_entropy(
                        logits.float(), batch["labels"].to(device)
                    )
                    / a.accum
                )
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * a.accum * batch["labels"].numel()
            seen += batch["labels"].numel()
            if step % a.accum == 0 or step == len(loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
        peak = (
            torch.cuda.max_memory_allocated()
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        record = {
            "epoch": epoch,
            "train_loss": running / seen,
            "pairs": seen,
            "seconds": round(time.time() - began, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "peak_device_bytes": peak,
        }
        history.append(record)
        print("epoch " + json.dumps(record, sort_keys=True), flush=True)
        if a.keep_epochs:
            keep = out / f"epoch-{epoch}"
            model.save_pretrained(keep, safe_serialization=True)
            tokenizer.save_pretrained(keep)
            for stale in sorted(out.glob("epoch-*"))[: -a.keep_epochs]:
                try:
                    shutil.rmtree(stale)
                except OSError as error:
                    print(f"warning: could not prune {stale}: {error}", flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    write(
        out / "lev-encoder.json",
        {
            "engine": "encoder",
            "backbone": {"repo": a.model, "revision": a.revision},
            "max_len": a.max_len,
            "labels": ["false", "true"],
            "training": {
                "epochs": a.epochs,
                "lr": a.lr,
                "batch": a.batch,
                "accum": a.accum,
                "dtype": a.dtype,
                "seed": a.seed,
                "warmup": a.warmup,
            },
            "data": {"path": str(Path(a.data).resolve()), "pairs": len(items)},
            "history": history,
            "seconds": round(time.time() - started, 2),
            "status": "trained-not-validated",
        },
    )
    print(f"wrote {out} in {time.time() - started:.1f}s", flush=True)


def predict(a):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    device = torch.device(a.device)
    checkpoint = Path(a.checkpoint)
    try:
        meta = json.loads((checkpoint / "lev-encoder.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{checkpoint}: not an encoder checkpoint ({error})"
        ) from error
    if a.max_len != meta["max_len"]:
        raise ValueError(
            f"--max-len {a.max_len} differs from the checkpoint's {meta['max_len']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    try:
        model = load_model(checkpoint, device).eval()
    except (OSError, ValueError) as error:
        raise ValueError(f"{checkpoint}: cannot load the encoder ({error})") from error
    items = load(a.data)
    ids, masks = tokenize(tokenizer, items, a.max_len)
    groups = order([len(x) for x in ids], a.batch, a.max_tokens, 0)
    loader = DataLoader(
        dataset(items, ids, masks),
        batch_sampler=groups,
        num_workers=0,
        collate_fn=collate(tokenizer),
    )
    scores: list[list[float]] = [[] for _ in range(len(items))]
    with torch.no_grad():
        for indices, batch in zip(groups, loader):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.float()
            for row, index in enumerate(indices):
                scores[index] = torch.softmax(logits[row], -1).cpu().tolist()
    unscored = [i for i, score in enumerate(scores) if len(score) != 2]
    if unscored:
        raise RuntimeError(
            f"scored {len(items) - len(unscored)} of {len(items)} pairs; first missing {unscored[0]}"
        )
    rows = []
    for item, score in zip(items, scores):
        total = score[0] + score[1]
        rows.append(
            {
                "id": item["id"],
                "group": item["group"],
                "question": item["question"],
                "source": item["source"],
                "task": item["kind"],
                "type": "noul",
                "variant": item["variant"],
                "keys": ["false", "true"],
                "label": item["label"],
                "p": [score[0] / total, score[1] / total],
            }
        )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write("[\n")
        for index, row in enumerate(rows):
            comma = "," if index + 1 < len(rows) else ""
            stream.write("  " + encoded(row) + comma + "\n")
        stream.write("]\n")
    print(f"wrote {len(rows)} predictions to {out}", flush=True)


def main():
    a = parse()
    if a.mode == "train":
        train(a)
    else:
        predict(a)


if __name__ == "__main__":
    main()
