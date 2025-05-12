import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from transformers import T5TokenizerFast
from tqdm import tqdm

from evaluator.toolkit import *
from memoizer import memoize

Tokenizer = T5TokenizerFast.from_pretrained("t5-small")


def get_string_length(s: str):
    return len(s)


def get_token_length(s: str):
    return len(Tokenizer.encode(s))


def argeq(a, b):
    return a[0] == b[0]


@memoize(Path("cache", "dataset_str_length_cache.pt"), arg_eq=argeq)
def get_data_tok_length(
    dataset_name: Tp.Literal["Human", "ChatGPT", "Claude", "Gemini"],
):
    selected_files = {
        "Human": Human_Data,
        "ChatGPT": GPT4_Data,
        "Claude": Claude_Data,
        "Gemini": Gemini_Data,
    }[dataset_name]
    all_data = load_data(selected_files)
    return [get_token_length(s) for s in tqdm(all_data)]


def plot_data_length_distribution():
    human_len = get_data_tok_length("Human")
    gpt4_len = get_data_tok_length("ChatGPT")
    claude_len = get_data_tok_length("Claude")
    gemini_len = get_data_tok_length("Gemini")

    colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e"]
    categories = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]
    all_data = [human_len, gpt4_len, claude_len, gemini_len]

    fig, axes = plt.subplots(ncols=1, nrows=4, dpi=200, sharey=True)
    for idx, (category, data, color, ax) in enumerate(
        zip(categories, all_data, colors, axes)
    ):
        ax.hist(data, bins=100, range=(0, 2500), color=color, density=True)
        ax.grid(visible=True, linestyle="--")
        ax.set_ylim(0, 0.005)

    # axes[-1].set_xlabel("Sample Length (# Token)")
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    for ax in axes:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    handles = [
        mpatches.Patch(color=c, label=label) for c, label in zip(colors, categories)
    ]
    fig: plt.Figure
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), handles=handles)
    fig.tight_layout()

    fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.text(0.01, 0.5, "Frequency", va="center", rotation="vertical")
    fig.text(0.5, 0.01, "Sample Length in our collected dataset (# Token)", ha="center")
    fig.savefig("./result/data/dataset_length_token.pdf")


def plot_data_length_distribution_cut():
    human_len = get_data_tok_length("Human")
    gpt4_len = get_data_tok_length("ChatGPT")
    claude_len = get_data_tok_length("Claude")
    gemini_len = get_data_tok_length("Gemini")

    human_len = [min(l, 512) for l in human_len]
    gpt4_len = [min(l, 512) for l in gpt4_len]
    claude_len = [min(l, 512) for l in claude_len]
    gemini_len = [min(l, 512) for l in gemini_len]

    colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e"]
    categories = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]
    all_data = [human_len, gpt4_len, claude_len, gemini_len]

    fig, axes = plt.subplots(ncols=1, nrows=4, dpi=200, sharey=True)
    for idx, (category, data, color, ax) in enumerate(
        zip(categories, all_data, colors, axes)
    ):
        ax.hist(data, bins=100, range=(0, 512), color=color, density=True)
        ax.grid(visible=True, linestyle="--")
        ax.set_ylim(0, 0.15)

    # axes[-1].set_xlabel("Sample Length seen by T5-Sentinel (# Token)")
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    for ax in axes:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    handles = [
        mpatches.Patch(color=c, label=label) for c, label in zip(colors, categories)
    ]
    fig: plt.Figure
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), handles=handles)
    fig.tight_layout()

    fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.text(0.01, 0.5, "Frequency", va="center", rotation="vertical")
    fig.text(0.5, 0.01, "Sample Length received by T5-Sentinel (# Token)", ha="center")
    fig.savefig("./result/data/dataset_length_token_cut.pdf")


if __name__ == "__main__":
    TASKS = [plot_data_length_distribution, plot_data_length_distribution_cut]
    for task in TASKS:
        print(f"Executing {task.__name__}")
        task()
