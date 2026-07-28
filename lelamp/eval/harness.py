from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

CLIPS_DIR = Path("eval_data/clips")


class EngagementMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    false_engagement_rate_per_hour: float
    mean_time_to_engage_s: float
    mean_time_to_disengage_s: float


class MemoryQAResult(BaseModel):
    exact_match_rate: float
    spatial_accuracy_within_20cm_rate: float


class EvalReport(BaseModel):
    engagement: EngagementMetrics
    memory_qa: MemoryQAResult
    slices: dict[str, EngagementMetrics]


def run_engagement_eval(clips_dir: Path = CLIPS_DIR) -> EngagementMetrics:
    raise NotImplementedError


def run_memory_qa_eval(qa_set_path: Path) -> MemoryQAResult:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
