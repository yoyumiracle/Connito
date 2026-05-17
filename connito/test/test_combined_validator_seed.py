"""Regression tests for `get_combined_validator_seed`.

The seed is derived by mixing TWO entropy sources:
- validator-committed `miner_seed` values (existing, backward-compat
  scheme; readable by miners during their commit window)
- the block hash of the last block of MinerCommit2 (new, added in
  the block-hash-mix-in PR — sealed before miners commit but
  unpredictable in advance)

If block_hash is unavailable (phase API or chain RPC transient
failure), it falls back to an empty string so the validator can
continue through the cycle. A follow-up PR will harden the
combined "no committed seeds AND no block_hash" case which currently
produces sha256("") — a publicly-known constant.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

from connito.shared.chain import ValidatorChainCommit
from connito.shared.cycle import PhaseNames, get_combined_validator_seed


# A stable, made-up block hash for tests. Real block hashes are
# 0x-prefixed 64-hex strings; matching that format ensures we exercise
# the same hashing code path.
TEST_BLOCK_HASH = "0xabcdef0123456789" + ("0" * 48)
TEST_PHASE_END_BLOCK = 8_184_162


class _StubSubtensor:
    """Minimal subtensor stand-in. The `get_block_hash` attribute is
    populated per-test via patching `_get_minercommit2_block_hash`."""

    def get_block_hash(self, block: int) -> str:
        return TEST_BLOCK_HASH


def _config_for_group(group_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=group_id)))


def _validator_commit(miner_seed: int | None, expert_group: int = 0) -> ValidatorChainCommit:
    return ValidatorChainCommit(miner_seed=miner_seed, expert_group=expert_group)


def _neuron(hotkey: str) -> SimpleNamespace:
    return SimpleNamespace(hotkey=hotkey)


# Patches the block-hash helper so tests don't actually hit the phase
# API or a live subtensor. Returning None simulates RPC failure; a
# string simulates a successful chain query.
def _patch_block_hash(value: str | None):
    return patch(
        "connito.shared.cycle._get_minercommit2_block_hash",
        return_value=value,
    )


def test_combined_seed_mixes_validator_seeds_and_block_hash():
    """Happy path: combined seed is sha256 of (sorted validator seeds
    concatenated) + block_hash. Both inputs contribute."""
    commits = [
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_z")),
        (_validator_commit(miner_seed=7, expert_group=0), _neuron("hk_a")),
        (_validator_commit(miner_seed=99, expert_group=0), _neuron("hk_m")),
    ]
    with _patch_block_hash(TEST_BLOCK_HASH):
        out = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=commits,
        )

    # Hotkey-sorted order: hk_a=7, hk_m=99, hk_z=42 → "79942"
    # Then concat block_hash, then sha256.
    expected = hashlib.sha256(("79942" + TEST_BLOCK_HASH).encode()).hexdigest()
    assert out == expected


def test_combined_seed_uses_only_block_hash_when_no_validator_seeds():
    """Transition path: during partial rollout (or if no validators
    publish miner_seed), the block-hash component alone determines
    the combined seed. This is still secure because block_hash isn't
    predictable in advance."""
    with _patch_block_hash(TEST_BLOCK_HASH):
        out = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=[],
        )

    expected = hashlib.sha256(("" + TEST_BLOCK_HASH).encode()).hexdigest()
    assert out == expected


def test_combined_seed_falls_back_to_empty_when_block_hash_unavailable():
    """If the block-hash component cannot be derived, fall back to
    using an empty string for that component so the validator keeps
    running through transient phase-API / chain-RPC outages.

    During the fallback window the combined seed is sha256(committed_part)
    only — same security level as the pre-PR scheme, which miners CAN
    predict from chain reads. Acceptable as a brief transient.

    NOTE: a follow-up PR will harden the combined "no committed seeds
    AND no block_hash" case, which currently produces sha256("") (a
    publicly-known constant). That gap is intentionally left open in
    this PR.
    """
    commits = [
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_a")),
    ]
    with _patch_block_hash(None):
        out = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=commits,
        )

    # block_hash falls back to "", combined = sha256("42" + "") = sha256("42")
    expected = hashlib.sha256(b"42").hexdigest()
    assert out == expected


def test_combined_seed_filters_by_expert_group():
    """Validators commit per-expert-group; seeds from other groups
    must not contaminate this group's combined seed."""
    commits = [
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_a")),
        (_validator_commit(miner_seed=999, expert_group=1), _neuron("hk_b")),  # wrong group
    ]
    with _patch_block_hash(TEST_BLOCK_HASH):
        out = get_combined_validator_seed(
            _config_for_group(group_id=0), _StubSubtensor(), commits=commits,
        )

    expected = hashlib.sha256(("42" + TEST_BLOCK_HASH).encode()).hexdigest()
    assert out == expected


def test_combined_seed_changes_when_block_hash_changes():
    """Different block hashes → different combined seeds. This is the
    cycle-over-cycle rotation that makes the eval data slice fresh."""
    commits = [
        (_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_a")),
    ]
    with _patch_block_hash(TEST_BLOCK_HASH):
        seed_a = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=commits,
        )

    other_block_hash = "0xdeadbeef" + ("0" * 56)
    with _patch_block_hash(other_block_hash):
        seed_b = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(), commits=commits,
        )

    assert seed_a != seed_b


def test_combined_seed_changes_when_validator_seeds_change():
    """Different validator seeds → different combined seeds. This
    preserves the original entropy source's contribution during the
    transition period when both components are mixed."""
    with _patch_block_hash(TEST_BLOCK_HASH):
        seed_a = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(),
            commits=[(_validator_commit(miner_seed=42, expert_group=0), _neuron("hk_a"))],
        )
        seed_b = get_combined_validator_seed(
            _config_for_group(), _StubSubtensor(),
            commits=[(_validator_commit(miner_seed=43, expert_group=0), _neuron("hk_a"))],
        )

    assert seed_a != seed_b
