from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

# Metadata (HEAD) request timeout passed explicitly to `hf_hub_download` so a
# stuck etag lookup can't extend our wall-clock budget. HF's chunk-fetch
# timeout is controlled separately via the `HF_HUB_DOWNLOAD_TIMEOUT` env var
# (default 10s) — we don't override that here so operators can tune it.
_HF_ETAG_TIMEOUT_SEC = 10.0


def _resolve_token(token: str | None, env_var: str) -> str | None:
    if token:
        return token
    return os.environ.get(env_var) or None


def resolve_hf_token(token: str | None = None, token_env_var: str = "HF_TOKEN") -> str | None:
    return _resolve_token(token, token_env_var)


@lru_cache(maxsize=8)
def _resolve_default_checkpoint_repo_from_token(resolved_token: str, default_repo_name: str) -> str | None:
    api = HfApi(token=resolved_token)
    whoami = api.whoami()
    namespace = str(whoami.get("name") or "").strip()
    if not namespace:
        logger.warning("HF whoami did not return a namespace for default checkpoint repo derivation")
        return None
    return f"{namespace}/{default_repo_name}"


def resolve_default_checkpoint_repo(
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    default_repo_name: str = "co",
) -> str | None:
    repo_name = default_repo_name.strip()
    if not repo_name:
        raise ValueError("default_repo_name cannot be empty")
    if "/" in repo_name:
        raise ValueError("default_repo_name must be a repo name, not '<namespace>/<repo>'")

    resolved_token = _resolve_token(token, token_env_var)
    if resolved_token is None:
        return None

    try:
        return _resolve_default_checkpoint_repo_from_token(resolved_token, repo_name)
    except Exception as exc:
        logger.warning(
            "Failed to derive default HF checkpoint repo from authenticated user",
            default_repo_name=repo_name,
            error=str(exc),
        )
        return None


def get_hf_upload_readiness(
    repo_id: str | None,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
) -> tuple[bool, str]:
    if not repo_id:
        return False, "HF checkpoint repo not configured"
    if _resolve_token(token, token_env_var) is None:
        return False, f"HF token missing — set {token_env_var} or pass token= explicitly"
    return True, "ready"


def resolve_hf_repo_ids(
    hf_cfg,
    max_chain_repo_chars: int | None = None,
) -> tuple[str | None, str | None]:
    """Resolve (upload_repo_id, chain_repo_id) from an HfCfg-shaped object.

    Shared between validator and miner so the derivation rules are identical:
    honor the explicit ``checkpoint_repo`` if set, otherwise derive
    ``{authenticated_hf_user}/{default_repo_name}`` from the HF token.
    ``max_chain_repo_chars`` enforces the role-specific chain payload budget
    (validators have a tighter one than miners).
    """
    derived_repo = resolve_default_checkpoint_repo(
        token_env_var=hf_cfg.token_env_var,
        default_repo_name=hf_cfg.default_repo_name,
    )
    upload_repo_id = hf_cfg.resolve_upload_repo(derived_repo)
    chain_repo_id = hf_cfg.advertised_repo_id(upload_repo_id)

    if chain_repo_id and max_chain_repo_chars is not None and len(chain_repo_id) > max_chain_repo_chars:
        raise ValueError(
            "HF chain repo id is too long for the chain payload budget: "
            f"{len(chain_repo_id)} > {max_chain_repo_chars}"
        )

    return upload_repo_id, chain_repo_id


def upload_checkpoint_to_hf(
    ckpt_dir: Path,
    repo_id: str,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    commit_message: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Upload a checkpoint directory to HF and return a short commit revision.

    The returned value is the first 12 chars of the commit SHA so callers can
    use a shorter immutable revision token downstream. `main` is updated to
    point at the new commit as a side effect, but miners should pin to the
    revision from the chain, not the branch name.
    """
    ready, reason = get_hf_upload_readiness(repo_id=repo_id, token=token, token_env_var=token_env_var)
    if not ready:
        raise RuntimeError(reason)
    resolved_token = _resolve_token(token, token_env_var)
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    api = HfApi(token=resolved_token)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    except HfHubHTTPError as e:
        # 409 on already-exists races is fine; anything else is real.
        if getattr(e.response, "status_code", None) not in (409,):
            raise

    # Default uploads expert-group shards only (both `.pt` and `.safetensors`
    # during the migration window). `model_shared.*` is intentionally excluded
    # from the default: it's no longer persisted or distributed; every
    # participant reconstructs backbone state from `config.model.model_path`
    # at startup. Callers that need different behavior can override
    # `allow_patterns` explicitly.
    default_allow_patterns = [
        "model_expgroup_*.pt",
        "model_expgroup_*.safetensors",
    ]
    commit_info = api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        commit_message=commit_message or f"checkpoint upload from {ckpt_dir.name}",
        allow_patterns=allow_patterns if allow_patterns is not None else default_allow_patterns,
    )
    revision = commit_info.oid[:12]
    logger.info(
        "Uploaded checkpoint to HF",
        repo_id=repo_id,
        revision=revision,
        src_dir=str(ckpt_dir),
    )
    return revision


def download_checkpoint_from_hf(
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: Path,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
) -> Path:
    """Download specific files from a HF repo revision into dest_dir.

    We download only the shards the caller needs (e.g. `model_expgroup_3.pt`)
    rather than the whole repo, since a validator may publish every expert
    group and a given miner only needs one. `model_shared.*` is no longer
    distributed; backbone state is reconstructed from
    `config.model.model_path` at instantiation.
    """
    resolved_token = _resolve_token(token, token_env_var)
    dest_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[str] = []
    missing: list[str] = []
    try:
        for fname in filenames:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    revision=revision,
                    filename=fname,
                    local_dir=str(dest_dir),
                    token=resolved_token,
                    etag_timeout=_HF_ETAG_TIMEOUT_SEC,
                )
                fetched.append(fname)
            except EntryNotFoundError:
                # Caller passes both `.safetensors` and `.pt` variants per
                # expert group during the format migration; tolerate the
                # absent variant and require at least one sibling to land.
                missing.append(fname)
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"HF repo not found or unauthorized: {repo_id}@{revision}"
        ) from e

    if not fetched:
        raise RuntimeError(
            f"No requested files present on HF {repo_id}@{revision}: missing={missing}"
        )

    logger.debug(
        "Downloaded checkpoint from HF",
        repo_id=repo_id,
        revision=revision,
        files=fetched,
        missing=missing,
        dest_dir=str(dest_dir),
    )
    return dest_dir


def download_checkpoint_from_hf_with_timeout(
    *,
    repo_id: str,
    revision: str,
    filenames: list[str],
    dest_dir: Path,
    token: str | None = None,
    token_env_var: str = "HF_TOKEN",
    timeout_sec: float | None,
) -> Path:
    """Wall-clock-bounded variant of `download_checkpoint_from_hf`.

    `asyncio.wait_for(asyncio.to_thread(...))` cannot interrupt a thread
    blocked inside `huggingface_hub` (it uses requests/httpx under the hood),
    so a hung download leaks a worker into the asyncio default executor pool
    indefinitely. Once enough leak, every subsequent `asyncio.to_thread`
    call queues behind a zombie and times out without ever sending bytes.

    Instead, run each attempt in its own one-shot ThreadPoolExecutor. On
    timeout we `shutdown(wait=False, cancel_futures=True)` so the zombie
    detaches from this caller; the underlying thread continues until its OS
    socket eventually closes, but it no longer blocks the asyncio loop and
    no longer starves unrelated `to_thread` callers.
    """
    if timeout_sec is None:
        return download_checkpoint_from_hf(
            repo_id=repo_id,
            revision=revision,
            filenames=filenames,
            dest_dir=dest_dir,
            token=token,
            token_env_var=token_env_var,
        )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-dl")
    future = executor.submit(
        download_checkpoint_from_hf,
        repo_id=repo_id,
        revision=revision,
        filenames=filenames,
        dest_dir=dest_dir,
        token=token,
        token_env_var=token_env_var,
    )
    try:
        return future.result(timeout=timeout_sec)
    finally:
        # wait=False: a hung thread won't be joined — we accept the leak so
        # the caller can move on. cancel_futures=True clears anything queued
        # (no-op here since max_workers=1 and we submitted one task).
        executor.shutdown(wait=False, cancel_futures=True)
