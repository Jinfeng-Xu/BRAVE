"""Tool registry and embedding-path helpers for BRAVE.

The repository does not ship trained recommender checkpoints. To reproduce a
paper setting, train or download tool embeddings and place them under one of
the searched directories, or pass ``--embedding_root`` to point to a custom
folder containing files named ``{tool}_{dataset}_{kind}_embeddings.pt``.
"""

from pathlib import Path


TOOL_INFO = {
    "lgn": "a graph-based collaborative filtering model",
    "lightgcn": "a graph-based collaborative filtering model",
    "srec": "an attention-based sequential recommendation model",
    "sasrec": "an attention-based sequential recommendation model",
    "smpx": "a matching-based collaborative filtering model",
    "simplex": "a matching-based collaborative filtering model",
    "mf": "a matrix-factorization collaborative filtering model",
    "enmf": "an efficient neural matrix-factorization recommender",
    "diffrec": "a diffusion-based recommender",
    "lrurec": "a linear recurrent sequential recommender",
    "fearec": "a frequency-enhanced sequential recommender",
    "wgcl": "a graph collaborative filtering model with weighted graph contrastive learning",
    "nlgcl": "a graph collaborative filtering model with neighborhood-level graph contrastive learning",
}


TOOL_POOLS = {
    "simple": ["lgn", "srec", "smpx"],
    "ultra": ["nlgcl", "wgcl", "fearec"],
}


TOOL_ALIASES = {
    "lightgcn": "lgn",
    "sasrec": "srec",
    "simplex": "smpx",
}


DEFAULT_EMBEDDING_DIRS = {
    "lgn": ["LightGCN-PyTorch/code/saved_embeddings"],
    "lightgcn": ["LightGCN-PyTorch/code/saved_embeddings"],
    "mf": ["LightGCN-PyTorch/code/saved_embeddings"],
    "srec": ["SASRec.pytorch/saved_embeddings"],
    "sasrec": ["SASRec.pytorch/saved_embeddings"],
    "smpx": ["SimpleX/saved_embeddings"],
    "simplex": ["SimpleX/saved_embeddings"],
}


def expand_tool_pool(tools):
    """Expand named pools while preserving arbitrary user-provided tools."""
    expanded = []
    for tool in tools:
        key = TOOL_ALIASES.get(tool.lower(), tool.lower())
        expanded.extend(TOOL_POOLS.get(key, [key]))
    return expanded


def tool_description(tool):
    return TOOL_INFO.get(tool.lower(), f"the recommendation tool named {tool}")


def embedding_path(tool, dataset, kind, embedding_root=None):
    """Return the first matching embedding path for a tool.

    Parameters
    ----------
    tool: str
        Tool name such as ``lgn`` or ``nlgcl``.
    dataset: str
        Dataset name used in the embedding filename.
    kind: {"user", "item"}
        Embedding kind.
    embedding_root: str | None
        Optional custom root. If provided, it is searched before defaults.
    """
    tool = TOOL_ALIASES.get(tool.lower(), tool.lower())
    filename = f"{tool}_{dataset}_{kind}_embeddings.pt"
    candidates = []
    if embedding_root:
        candidates.append(Path(embedding_root) / filename)
        candidates.append(Path(embedding_root) / tool / filename)
    for root in DEFAULT_EMBEDDING_DIRS.get(tool, []):
        candidates.append(Path(root) / filename)
    candidates.append(Path("embeddings") / filename)
    candidates.append(Path("embeddings") / tool / filename)
    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Missing {kind} embeddings for tool={tool}, dataset={dataset}. "
        f"Expected {filename}; searched: {searched}"
    )
