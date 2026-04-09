from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class AutoKGConfig:
    repo_root: Path
    project_root: Path
    env_path: Path
    knowledgegraph_path: Path
    title_knowledgegraph_path: Path
    replace_path: Path
    replace_title_path: Path
    user_dict_path: Path
    data_dir: Path
    reports_dir: Path
    prompts_dir: Path
    schemas_dir: Path
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout: int
    llm_max_tokens: int
    llm_batch_size: int
    llm_confidence_threshold: float


def load_config() -> AutoKGConfig:
    project_root = Path(__file__).resolve().parent
    repo_root = project_root.parent.parent
    env_path = repo_root / ".env"
    env_values = _load_env_file(env_path)

    return AutoKGConfig(
        repo_root=repo_root,
        project_root=project_root,
        env_path=env_path,
        knowledgegraph_path=repo_root / "config" / "knowledgegraph.xlsx",
        title_knowledgegraph_path=repo_root / "config" / "knowledgegraph_title.xlsx",
        replace_path=repo_root / "config" / "replace.xlsx",
        replace_title_path=repo_root / "config" / "replace_title.xlsx",
        user_dict_path=repo_root / "config" / "user_dic_expand.txt",
        data_dir=project_root / "data",
        reports_dir=project_root / "reports",
        prompts_dir=project_root / "prompts",
        schemas_dir=project_root / "schemas",
        llm_base_url=env_values.get("LLM_BASE_URL", os.getenv("LLM_BASE_URL", "")),
        llm_model=env_values.get("LLM_MODEL", os.getenv("LLM_MODEL", "")),
        llm_api_key=env_values.get("LLM_API_KEY", os.getenv("LLM_API_KEY", "")),
        llm_timeout=int(env_values.get("LLM_TIMEOUT", os.getenv("LLM_TIMEOUT", "30"))),
        llm_max_tokens=int(env_values.get("LLM_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "2048"))),
        llm_batch_size=int(env_values.get("LLM_BATCH_SIZE", os.getenv("LLM_BATCH_SIZE", "5"))),
        llm_confidence_threshold=float(
            env_values.get("LLM_CONFIDENCE_THRESHOLD", os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.7"))
        ),
    )


def ensure_runtime_dirs(cfg: AutoKGConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.prompts_dir.mkdir(parents=True, exist_ok=True)
    cfg.schemas_dir.mkdir(parents=True, exist_ok=True)