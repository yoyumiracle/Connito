#!/usr/bin/env python3
"""Fetch validator chain commits and verify they're distributing models.

For each whitelisted validator on subnet 102, shows:
  - their UID, hotkey
  - last_update age in blocks (>= cycle_length means weights are stale)
  - committed (global_ver, expert_group, model_hash, hf_repo_id, hf_revision)
  - HF reachability (optional, --check-hf): does the repo+revision exist?
  - validity verdict

One-shot:
    python scripts/poll_validators.py
With HF reachability checks:
    python scripts/poll_validators.py --check-hf
Continuous:
    python scripts/poll_validators.py --watch 600 --check-hf

Appends each fetch to scripts/validator_log.jsonl.
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
DEFAULT_NETWORK = "finney"
CYCLE_LENGTH_DEFAULT = 448  # from config.yaml cycle.cycle_length
WHITELIST_PATH = pathlib.Path(__file__).resolve().parents[1] / "connito" / "sn_owner" / "validator_whitelist.json"
LOG_PATH = pathlib.Path(__file__).resolve().parent / "validator_log.jsonl"

# Compact field aliases used in ValidatorChainCommit JSON on chain.
# Source: connito/shared/chain.py:61-72
ALIAS_MAP = {
    "h":  "model_hash",
    "v":  "global_ver",
    "e":  "expert_group",
    "s":  "miner_seed",
    "r":  "hf_repo_id",
    "rv": "hf_revision",
    "m":  "signed_model_hash",
    "b":  "block",
    "i":  "inner_opt",
}


def _expand(d: dict) -> dict:
    return {ALIAS_MAP.get(k, k): v for k, v in d.items()}


def _load_whitelist() -> set[str]:
    with WHITELIST_PATH.open() as f:
        return set(json.load(f))


def _check_hf(repo_id: str, revision: str | None) -> tuple[bool, str]:
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        info = api.repo_info(repo_id=repo_id, revision=revision, repo_type="model")
        sha = (getattr(info, "sha", "") or "")[:7]
        return True, f"ok (sha={sha}, lastModified={getattr(info,'lastModified','?')})"
    except Exception as e:
        return False, f"failed: {e.__class__.__name__}: {e}"


def fetch(netuid: int, network: str, check_hf: bool, cycle_length: int) -> dict:
    sub = bt.Subtensor(network=network)
    block = sub.get_current_block()
    mg = sub.metagraph(netuid=netuid, lite=True)
    commits = sub.get_all_commitments(netuid=netuid)

    hotkey_to_uid = {hk: uid for uid, hk in enumerate(mg.hotkeys)}
    whitelist = _load_whitelist()

    rows = []
    for hk in whitelist:
        uid = hotkey_to_uid.get(hk)
        record: dict = {"hotkey": hk, "uid": uid}

        if uid is None:
            record["status"] = "DEREGISTERED (not in metagraph)"
            rows.append(record)
            continue

        last_update = int(getattr(mg.neurons[uid], "last_update", 0)) if hasattr(mg, "neurons") else 0
        # fallback if neurons attr absent on this SDK version
        if last_update == 0:
            try:
                last_update = int(mg.last_update[uid])
            except Exception:
                pass
        age = block - last_update
        record["last_update_block"] = last_update
        record["age_blocks"] = age
        record["fresh"] = age <= cycle_length

        raw = commits.get(hk)
        if raw is None:
            record["status"] = "NO COMMIT"
            rows.append(record)
            continue

        try:
            parsed_raw = json.loads(raw) if isinstance(raw, str) else raw
            commit = _expand(parsed_raw)
        except Exception as e:
            record["status"] = f"COMMIT UNPARSEABLE: {e}"
            record["raw_preview"] = (raw if isinstance(raw, str) else str(raw))[:120]
            rows.append(record)
            continue

        record["commit"] = commit
        required = ("model_hash", "global_ver", "expert_group", "hf_repo_id", "hf_revision")
        missing = [k for k in required if commit.get(k) in (None, "")]
        record["missing_fields"] = missing

        if check_hf and not missing:
            ok, msg = _check_hf(commit["hf_repo_id"], commit["hf_revision"])
            record["hf_ok"] = ok
            record["hf_msg"] = msg

        if missing:
            record["status"] = f"INVALID (missing: {','.join(missing)})"
        elif not record["fresh"]:
            record["status"] = f"STALE WEIGHTS (age={age} > cycle={cycle_length})"
        elif check_hf and not record.get("hf_ok", True):
            record["status"] = "HF UNREACHABLE"
        else:
            record["status"] = "OK"

        rows.append(record)

    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "block": int(block),
        "netuid": netuid,
        "validators": rows,
    }


def _print(snap: dict) -> None:
    print(f"\n[{snap['ts']}] block={snap['block']}  netuid={snap['netuid']}  "
          f"({len(snap['validators'])} whitelisted validators)")
    print("-" * 100)
    status_counts: dict[str, int] = {}
    for r in snap["validators"]:
        uid_str = str(r.get("uid", "—")).rjust(4)
        age = r.get("age_blocks", "—")
        hk = r["hotkey"][:10] + "…"
        status = r["status"]
        head = status.split(" ", 1)[0]
        status_counts[head] = status_counts.get(head, 0) + 1

        commit = r.get("commit") or {}
        gver = commit.get("global_ver", "—")
        egrp = commit.get("expert_group", "—")
        repo = commit.get("hf_repo_id", "—")
        rev = commit.get("hf_revision", "—")
        mhash = (commit.get("model_hash") or "—")[:10]

        line = (f"  uid={uid_str}  {hk}  age={age:>5}  gver={gver:>4}  "
                f"egrp={egrp}  hash={mhash}…  hf={repo}@{rev}")
        print(line)
        if r.get("hf_msg"):
            print(f"        hf: {r['hf_msg']}")
        print(f"        => {status}")
    print("-" * 100)
    print("summary:", "  ".join(f"{k}:{v}" for k, v in sorted(status_counts.items())))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    p.add_argument("--network", default=DEFAULT_NETWORK, help="finney | archive | <ws_url>")
    p.add_argument("--cycle-length", type=int, default=CYCLE_LENGTH_DEFAULT,
                   help=f"blocks before weights count as stale (default: {CYCLE_LENGTH_DEFAULT})")
    p.add_argument("--check-hf", action="store_true",
                   help="Verify each validator's hf_repo_id@hf_revision is reachable")
    p.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    p.add_argument("--log", default=str(LOG_PATH))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    log_path = pathlib.Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def once() -> None:
        snap = fetch(args.netuid, args.network, args.check_hf, args.cycle_length)
        with log_path.open("a") as f:
            f.write(json.dumps(snap) + "\n")
        if not args.quiet:
            _print(snap)

    if args.watch <= 0:
        once()
        return

    while True:
        try:
            once()
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"fetch failed: {e}", file=sys.stderr)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
