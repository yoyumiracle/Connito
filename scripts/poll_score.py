#!/usr/bin/env python3
"""Poll your Connito miner score from the chain.

One-shot:
    python scripts/poll_score.py

Continuous (every 10 min):
    python scripts/poll_score.py --watch 600

Defaults match config.yaml: netuid=102, uid=174, network=finney.
Appends each sample to scripts/score_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

import bittensor as bt

DEFAULT_NETUID = 102
DEFAULT_UID = 174
DEFAULT_NETWORK = "finney"
LOG_PATH = pathlib.Path(__file__).resolve().parent / "score_log.jsonl"


def _weight_voters(mg, uid: int) -> list[tuple[int, float]]:
    """Return [(validator_uid, normalized_weight), ...] for validators voting on `uid`.

    Reads the full (non-lite) metagraph weights matrix: rows = voters, cols = miners.
    """
    import numpy as np

    W = np.asarray(mg.weights)
    if W.ndim != 2 or W.shape[1] <= uid:
        return []
    col = W[:, uid]
    voters = [(int(i), float(col[i])) for i in range(len(col)) if float(col[i]) > 0.0]
    voters.sort(key=lambda kv: -kv[1])
    return voters


def _hf_downloads(hf_repo_id: str | None) -> int | None:
    if not hf_repo_id:
        return None
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        info = HfApi(token=token).repo_info(repo_id=hf_repo_id, repo_type="model")
        return int(getattr(info, "downloads", 0) or 0)
    except Exception as e:
        print(f"  hf_downloads: failed ({e})", file=sys.stderr)
        return None


def _f(mg, name: str, uid: int) -> float:
    arr = getattr(mg, name, None)
    if arr is None:
        return float("nan")
    try:
        return float(arr[uid])
    except (IndexError, TypeError):
        return float("nan")


def sample(netuid: int, uid: int, network: str, hf_repo_id: str | None) -> dict:
    sub = bt.Subtensor(network=network)
    # lite=False is required to populate the weights matrix in bittensor >= 9.x
    mg = sub.metagraph(netuid=netuid, lite=False)

    n = int(getattr(mg, "n", len(mg.hotkeys)))
    if uid >= n:
        raise SystemExit(f"uid {uid} out of range (metagraph has {n} neurons)")

    voters = _weight_voters(mg, uid)
    block = sub.get_current_block()

    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "block": int(block),
        "uid": uid,
        "netuid": netuid,
        "hotkey": mg.hotkeys[uid],
        "incentive": _f(mg, "incentive", uid),
        "emission": _f(mg, "emission", uid),
        "stake": _f(mg, "stake", uid),
        "dividends": _f(mg, "dividends", uid),
        "validator_trust": _f(mg, "validator_trust", uid),
        "n_voters": len(voters),
        "weight_total": sum(w for _, w in voters),
        "voters": voters,
        "hf_downloads": _hf_downloads(hf_repo_id),
    }
    return row


def _verdict(row: dict) -> str:
    n = row["n_voters"]
    inc = row["incentive"]
    if inc > 0 and n >= 3:
        return "GROUP 1 (top-3, ~98% pool)"
    if inc > 0 and n >= 1:
        return "GROUP 2 (top-5, ~2% pool)"
    if row["hf_downloads"] is not None and row["hf_downloads"] == 0:
        return "REJECTED before eval (no HF downloads — likely signature/hash/upload issue)"
    return "NOT SCORED this cycle (mark_failed at eval, or not in foreground roster)"


def _print(row: dict) -> None:
    print(f"[{row['ts']}] block={row['block']}  uid={row['uid']}  hotkey={row['hotkey'][:10]}…")
    print(f"  incentive={row['incentive']:.6f}  emission={row['emission']:.6f}  "
          f"stake={row['stake']:.4f}  dividends={row['dividends']:.6f}")
    print(f"  voters={row['n_voters']}  total_weight={row['weight_total']:.4f}")
    if row["voters"]:
        top = ", ".join(f"v{u}:{w:.4f}" for u, w in row["voters"][:6])
        print(f"  top voters: {top}")
    if row["hf_downloads"] is not None:
        print(f"  hf_downloads (cumulative): {row['hf_downloads']}")
    print(f"  >> {_verdict(row)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    p.add_argument("--uid", type=int, default=DEFAULT_UID)
    p.add_argument("--network", default=DEFAULT_NETWORK, help="finney | archive | <ws_url>")
    p.add_argument("--hf-repo", default=os.environ.get("MINER_HF_REPO"),
                   help="Your miner HF repo id, e.g. user/repo. Or set MINER_HF_REPO env.")
    p.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                   help="Poll forever every N seconds (default: one-shot)")
    p.add_argument("--log", default=str(LOG_PATH), help=f"JSONL output (default: {LOG_PATH})")
    p.add_argument("--quiet", action="store_true", help="Only write JSONL, no stdout")
    args = p.parse_args()

    log_path = pathlib.Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def once() -> None:
        row = sample(args.netuid, args.uid, args.network, args.hf_repo)
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if not args.quiet:
            _print(row)

    if args.watch <= 0:
        once()
        return

    while True:
        try:
            once()
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"poll failed: {e}", file=sys.stderr)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
