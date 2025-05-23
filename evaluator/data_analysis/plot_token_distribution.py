import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from transformers import T5TokenizerFast

import memoizer
from evaluator.toolkit import *
from tqdm import tqdm
import string

Tokenizer = T5TokenizerFast.from_pretrained("t5-small")


def count_character(s: str, counter: dict) -> None:
    tokens = Tokenizer.encode(s)
    for tok in tokens:
        if tok in counter:
            counter[tok] += 1
        else:
            counter[tok] = 1


def argeq(a, b):
    return a[0] == b[0]


@memoizer.memoize(Path("cache", "dataset_token_count_cache.pt"), arg_eq=argeq)
def count_dataset(
    dataset: Tp.Literal["Human", "ChatGPT", "Claude", "Gemini"],
) -> dict:
    selected_files = {
        "Human": Human_Data,
        "ChatGPT": GPT4_Data,
        "Claude": Claude_Data,
        "Gemini": Gemini_Data,
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
    key_arr.sort(key=lambda x: x[0], reverse=False)
    return [entry[1] for entry in key_arr]


def plot_distribution():
    human_counter = count_dataset("Human")
    gpt4_counter = count_dataset("ChatGPT")
    claude_counter = count_dataset("Claude")
    gemini_counter = count_dataset("Gemini")

    # selected_keys = merge_keys(
    #     get_top_k_chars(human_counter, 40),
    #     get_top_k_chars(gpt3_counter, 40),
    #     get_top_k_chars(palm_counter, 40),
    #     get_top_k_chars(llama_counter, 40),
    #     get_top_k_chars(gpt2_counter, 40)
    # )
    selected_keys = get_top_k_chars(human_counter, 40)

    human_counter = filter_dict(human_counter, selected_keys)
    gpt4_counter = filter_dict(gpt4_counter, selected_keys)
    claude_counter = filter_dict(claude_counter, selected_keys)
    gemini_counter = filter_dict(gemini_counter, selected_keys)
    selected_keys = sort_keys(human_counter, selected_keys)
    display_keys = [Tokenizer.decode(k) for k in selected_keys]

    axes: Tp.List[plt.Axes]
    fig, axes = plt.subplots(ncols=4, nrows=1, dpi=200, sharey=True, sharex=True)
    colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e"]
    categories = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]
    all_data = [human_counter, gpt4_counter, claude_counter, gemini_counter]

    for idx, (category, counter, color, ax) in enumerate(
        zip(categories, all_data, colors, axes)
    ):
        values = [counter[k] for k in selected_keys]
        ax.barh(display_keys, values, color=color)

    for ax in axes[1:]:
        ax.get_yaxis().set_visible(False)
    for ax, cate in zip(axes, categories):
        ax.set_title(cate)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].set_yticklabels(display_keys, fontdict=dict(fontsize=7))

    fig.text(
        0.01,
        0.5,
        "Most common tokens in our collected dataset",
        va="center",
        rotation="vertical",
    )
    fig.text(0.5, 0.01, "Frequency", ha="center")
    fig.savefig("./result/data/dataset_token_count.pdf")


if __name__ == "__main__":
    TASKS = [plot_distribution]
    for task in TASKS:
        print(f"Executing {task.__name__}")
        task()
