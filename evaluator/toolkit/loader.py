from pipeline import P, PipelineExecutor, utils
from pathlib import Path
import typing as Tp


Human_Data = [
    Path("data", "split", "Human", "train-dirty.jsonl"),
    Path("data", "split", "Human", "test-dirty.jsonl"),
    Path("data", "split", "Human", "valid-dirty.jsonl"),
]

GPT4_Data = [
    Path("data", "split", "GPT-4", "test-dirty.jsonl"),
    Path("data", "split", "GPT-4", "train-dirty.jsonl"),
    Path("data", "split", "GPT-4", "valid-dirty.jsonl"),
]

Claude_Data = [
    Path("data", "split", "Claude-Instant-v1", "train-dirty.jsonl"),
    Path("data", "split", "Claude-Instant-v1", "valid-dirty.jsonl"),
    Path("data", "split", "Claude-Instant-v1", "test-dirty.jsonl"),
]

Gemini_Data = [
    Path("data", "split", "Gemini-Pro", "train-dirty.jsonl"),
    Path("data", "split", "Gemini-Pro", "valid-dirty.jsonl"),
    Path("data", "split", "Gemini-Pro", "test-dirty.jsonl"),
]


def load_data(files: Tp.List[Path]) -> Tp.Sequence[str]:
    executor = PipelineExecutor(worker_num=min(len(files), 8))
    result = executor.sequential_mapreduce(
        map_fn=P.FromJsonStr()
        >> P.ToStr()
        >> P.ToSingletonList(input_type=Tp.Optional[str]),
        from_files=files,
        identity=[],
        reduce_fn=utils.reduce_list,
    )
    return result
