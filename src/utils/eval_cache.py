"""评估结果缓存工具"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


class EvaluationCache:
    """用于复用评估结果的简单文件缓存"""

    VERSION = 1

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        default_dir = Path(__file__).resolve().parents[2] / ".cache" / "evaluations"
        self.cache_dir = Path(cache_dir or default_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_text(text: str) -> str:
        """计算文本的 sha256"""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_checkpoints(checkpoints: list[str]) -> str:
        """计算检查项列表的哈希"""

        joined = "\n".join(checkpoints)
        return EvaluationCache.hash_text(joined)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def build_key(
        self,
        *,
        doc_hash: str,
        checkpoints_hash: str,
        checkpoint_count: int,
        runs: int,
        model_name: str,
        prompt_version: str | None,
        baseline_fingerprint: str | None,
        mode: str,
    ) -> str:
        """构建缓存键"""

        payload = {
            "baseline": baseline_fingerprint or "",
            "checkpoint_count": checkpoint_count,
            "checkpoints_hash": checkpoints_hash,
            "doc_hash": doc_hash,
            "model": model_name,
            "mode": mode,
            "prompt_version": prompt_version or "",
            "runs": runs,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def load(self, key: str) -> Dict[str, Any] | None:
        """读取缓存内容"""

        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return None
        if data.get("version") != self.VERSION:
            return None
        return data

    def save(self, key: str, payload: Dict[str, Any]) -> None:
        """写入缓存"""

        path = self._cache_path(key)
        payload["version"] = self.VERSION
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def delete(self, key: str) -> None:
        """删除指定的缓存文件"""

        path = self._cache_path(key)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass  # 忽略删除失败的情况
