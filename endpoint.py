from pathlib import Path
import evaluator.models.t5_sentinel.t5_pipeline as pipeline

text = """I follow some basic rules to keep myself clear of trouble on motorways and
find the game of high-speed chess curiously satisfying. About half of
motorway inhabitants play this game, and you can communicate with them
on a quasi-Masonic handshake level. The other half are physically present
but mentally abstaining. These people typically languish in the middle lane.
Sometimes I join their group as well, usually in average speed zones, where
my focus is obliterated by the need to strain at the needle lest it passes 53
mph."""

pipe = (
    pipeline.ExecuteT5(Path("./data/checkpoint/T5Sentinel.0613.pt"))
    >> pipeline.T5PredictToLogits()
)
result = pipe(
    {
        "uid": "001",
        "text": text,
        "extra": None,
    }
)
print(result)
