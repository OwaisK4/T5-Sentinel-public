import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

import string
import memoizer
from evaluator.toolkit import *
from tqdm import tqdm
import string


def count_character(s: str, counter: dict) -> None:
    for c in s:
        if c in counter:
            counter[c] += 1
        else:
            counter[c] = 1


def argeq(a, b):
    return a[0] == b[0]


@memoizer.memoize(Path("cache", "dataset_char_count_cache.pt"), arg_eq=argeq)
def count_dataset(
    dataset: Tp.Literal["Human", "GPT-4", "Claude-Instant-v1", "Gemini-Pro"],
) -> dict:
    selected_files = {
        "Human": Human_Data,
        "GPT-4": GPT4_Data,
        "Claude-Instant-v1": Claude_Data,
        "Gemini-Pro": Gemini_Data,
    }[dataset]

    counter = {c: 0 for c in string.printable}
    dataset = load_data(selected_files)
    for entry in tqdm(dataset):
        count_character(entry, counter)

    total_token = sum([counter[k] for k in counter])
    result = {k: counter[k] / total_token for k in counter}
    return result


def get_top_k_chars(counter: dict, k: int) -> list:
    kv_pair = [(counter[k], k) for k in counter]
    kv_pair.sort(key=lambda x: x[0], reverse=True)
    return [entry[1] for entry in kv_pair[:k]]


def filter_dict(counter: dict, keys: list) -> dict:
    resulted_dict = {}
    for k in keys:
        resulted_dict[k] = counter[k] if k in counter else 0
    return resulted_dict


def merge_keys(*arr_keys) -> list:
    set_keys = set()
    for keys in arr_keys:
        set_keys = set_keys.union(set(keys))
    return list(set_keys)


def sort_keys(counter, keys) -> list:
    key_arr = [(counter[key], key) for key in keys]
    key_arr.sort(key=lambda x: x[0], reverse=True)
    return [entry[1] for entry in key_arr]


def plot_distribution():
    human_counter = count_dataset("Human")
    gpt4_counter = count_dataset("GPT-4")
    claude_counter = count_dataset("Claude-Instant-v1")
    gemini_counter = count_dataset("Gemini-Pro")

    selected_keys = merge_keys(
        get_top_k_chars(human_counter, 40),
        get_top_k_chars(gpt4_counter, 40),
        get_top_k_chars(claude_counter, 40),
        get_top_k_chars(gemini_counter, 40),
    )

    human_counter = filter_dict(human_counter, selected_keys)
    gpt4_counter = filter_dict(gpt4_counter, selected_keys)
    claude_counter = filter_dict(claude_counter, selected_keys)
    gemini_counter = filter_dict(gemini_counter, selected_keys)
    selected_keys = sort_keys(human_counter, selected_keys)

    fig, axes = plt.subplots(ncols=1, nrows=4, dpi=200, sharey=True, sharex=True)
    colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e"]
    categories = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]
    all_data = [human_counter, gpt4_counter, claude_counter, gemini_counter]

    for idx, (category, counter, color, ax) in enumerate(
        zip(categories, all_data, colors, axes)
    ):
        values = [counter[k] for k in selected_keys]
        ax.bar(selected_keys, values, color=color)

    for ax in axes[:-1]:
        ax.get_xaxis().set_visible(False)
    for ax in axes:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax = axes[-1]
    handles = [
        mpatches.Patch(color=c, label=label) for c, label in zip(colors, categories)
    ]
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), ncol=1, handles=handles)

    fig.text(0.01, 0.5, "Frequency", va="center", rotation="vertical")
    fig.text(0.5, 0.01, "Most common character in our collected dataset", ha="center")
    fig.tight_layout()
    fig.savefig("./result/data/dataset_char_count.pdf")


def plot_punc_distribution():
    human_counter = count_dataset("Human")
    gpt4_counter = count_dataset("GPT-4")
    claude_counter = count_dataset("Claude-Instant-v1")
    gemini_counter = count_dataset("Gemini-Pro")

    punctuation_tok = [tok for tok in string.punctuation]

    human_counter = filter_dict(human_counter, punctuation_tok)
    gpt4_counter = filter_dict(gpt4_counter, punctuation_tok)
    claude_counter = filter_dict(claude_counter, punctuation_tok)
    gemini_counter = filter_dict(gemini_counter, punctuation_tok)
    selected_keys = sort_keys(human_counter, punctuation_tok)

    fig, axes = plt.subplots(ncols=1, nrows=4, dpi=200, sharey=True, sharex=True)
    colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e"]
    categories = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]
    all_data = [human_counter, gpt4_counter, claude_counter, gemini_counter]

    for idx, (category, counter, color, ax) in enumerate(
        zip(categories, all_data, colors, axes)
    ):
        values = [counter[k] for k in selected_keys]
        ax.bar(selected_keys, values, color=color)

    # for ax in axes[:-1]: ax.get_xaxis().set_visible(False)
    for ax in axes:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    handles = [
        mpatches.Patch(color=c, label=label) for c, label in zip(colors, categories)
    ]
    fig: plt.Figure
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), handles=handles)
    fig.tight_layout()
    # axes[1].legend(handles=handles)
    fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.text(0.00, 0.5, "Frequency", va="center", rotation="vertical")
    fig.text(0.5, 0.01, "Most common punctuation in our collected dataset", ha="center")

    fig.savefig("./result/data/dataset_punc_count.pdf")


if __name__ == "__main__":
    TASKS = [plot_distribution, plot_punc_distribution]
    for task in TASKS:
        print(f"Executing {task.__name__}")
        task()
