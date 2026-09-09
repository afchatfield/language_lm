"""Resolve model specifications from the Hugging Face hub without downloading weights.

The fertility analysis needs a model's vocabulary geometry and its total
parameter count, but not its weights -- which is fortunate, because five
candidates would be roughly 15GB. Configs and tokenizers are a few megabytes
each, and the parameter count is available from hub metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import model_info
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoConfig, AutoTokenizer

from langlm.tokenizers.fertility import ModelSpec, spec_from_config

#: Weight file suffixes, and how many bytes per parameter each dtype implies.
_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


class ModelUnavailableError(RuntimeError):
    """Raised when a candidate cannot be inspected, typically because it is gated."""


@dataclass
class LoadedCandidate:
    """A model spec paired with its tokenizer."""

    spec: ModelSpec
    tokenizer: object


def load_candidate(name: str, repo_id: str, note: str = "") -> LoadedCandidate:
    """Fetch a candidate's config, tokenizer, and parameter count.

    Args:
        name: Short display name.
        repo_id: Hugging Face repository id.
        note: Commentary carried into the report.

    Raises:
        ModelUnavailableError: if the repository is gated or otherwise
            unreachable. Callers are expected to record the reason and continue,
            so that one inaccessible candidate does not sink the whole table.
    """
    try:
        config = AutoConfig.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
    except (OSError, HfHubHTTPError, ValueError) as exc:
        raise ModelUnavailableError(_explain(repo_id, exc)) from exc

    params = resolve_param_count(repo_id)
    if params is None:
        raise ModelUnavailableError(
            f"Could not determine the parameter count for {repo_id} from hub metadata."
        )

    return LoadedCandidate(
        spec=spec_from_config(name=name, repo_id=repo_id, params=params, config=config, note=note),
        tokenizer=tokenizer,
    )


def resolve_param_count(repo_id: str) -> int | None:
    """Total parameter count, from hub metadata rather than the model card.

    Advertised model names are unreliable -- "Qwen3-1.7B" counts only
    non-embedding parameters and totals 2.03B -- and this analysis expresses
    savings as a fraction of the whole model, so the real number matters.

    Two sources, in order of preference:

    1. The hub's parsed ``safetensors`` metadata, which gives an exact count.
    2. The size of the weight files divided by the bytes-per-parameter implied
       by the config's dtype. Used for repositories that predate safetensors.

    Returns:
        The parameter count, or ``None`` if neither source is available.
    """
    try:
        info = model_info(repo_id, files_metadata=True)
    except HfHubHTTPError:
        return None

    safetensors = getattr(info, "safetensors", None)
    if safetensors and getattr(safetensors, "total", None):
        return int(safetensors.total)

    weight_bytes = sum(
        sibling.size or 0
        for sibling in info.siblings
        if sibling.rfilename.endswith((".safetensors", ".bin"))
        and "training_args" not in sibling.rfilename
    )
    if not weight_bytes:
        return None

    config = AutoConfig.from_pretrained(repo_id)
    # `torch_dtype` was renamed to `dtype` in transformers 5; accept either.
    dtype = str(getattr(config, "dtype", None) or getattr(config, "torch_dtype", "float16"))
    bytes_per_param = _DTYPE_BYTES.get(dtype.replace("torch.", ""), 2)
    return round(weight_bytes / bytes_per_param)


def _explain(repo_id: str, exc: Exception) -> str:
    message = str(exc)
    if "gated repo" in message or "401" in message:
        return (
            f"{repo_id} is gated. Accept the licence at https://huggingface.co/{repo_id} "
            f"and run `hf auth login`, then re-run the analysis."
        )
    return f"{repo_id} could not be loaded: {message.splitlines()[0]}"
