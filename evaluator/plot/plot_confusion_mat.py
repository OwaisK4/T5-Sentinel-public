import matplotlib.pyplot as plt
import seaborn as sns

from evaluator.toolkit import *
from pathlib import Path

import evaluator.models.t5_sentinel.t5_get_hidden_states as T5_Full


def plot_full_confusion_mat():
    predictions = T5_Full.evaluate_predictions(
        [
            Path("data", "split", "Human", "test-dirty.jsonl"),
            Path("data", "split", "GPT-4", "test-dirty.jsonl"),
            Path("data", "split", "Claude-Instant-v1", "test-dirty.jsonl"),
            Path("data", "split", "Gemini-Pro", "test-dirty.jsonl"),
            # Path("./data/split/open-llama-text/test-dirty.jsonl"),
            # Path("./data/split/gpt2-output/test-dirty.jsonl")
        ]
    )
    display_labels = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]

    matrix = calculate_confusion_matrix(
        predictions, ["Human", "ChatGPT", "Claude", "Gemini"]
    )
    fig: plt.Figure = plt.figure(dpi=200)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.imshow(
        matrix, cmap=sns.color_palette("crest_r", as_cmap=True), interpolation="nearest"
    )
    ax.set_xticks([_ for _ in range(len(display_labels))], display_labels)
    ax.set_yticks([_ for _ in range(len(display_labels))], display_labels)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            ax.text(
                j,
                i,
                format(int(matrix[i, j]), "d"),
                horizontalalignment="center",
                color=(
                    "white"
                    if matrix[i, j] < np.sum(matrix) / (len(display_labels) + 1)
                    else "black"
                ),
                fontsize="large",
            )
    # ax.set_title("Confusion Matrix for T5-Sentinel")
    fig.tight_layout()
    fig.savefig(Path("./result/t5_sentinel/confusion_mat_t5_full_dirty.pdf"))


def plot_full_confusion_mat_abalation(file_name: str, variant_level: int):
    def implement():
        predictions = T5_Full.evaluate_predictions(
            [
                Path("data", "split", "Human", file_name),
                Path("data", "split", "GPT-4", file_name),
                Path("data", "split", "Claude-Instant-v1", file_name),
                Path("data", "split", "Gemini-Pro", file_name),
            ]
        )
        display_labels = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]

        matrix = calculate_confusion_matrix(
            predictions, ["Human", "ChatGPT", "Claude", "Gemini"]
        )
        fig: plt.Figure = plt.figure(dpi=200)
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        ax.imshow(
            matrix,
            cmap=sns.color_palette("crest_r", as_cmap=True),
            interpolation="nearest",
        )
        ax.set_xticks([_ for _ in range(len(display_labels))], display_labels)
        ax.set_yticks([_ for _ in range(len(display_labels))], display_labels)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        for i in range(len(display_labels)):
            for j in range(len(display_labels)):
                ax.text(
                    j,
                    i,
                    format(int(matrix[i, j]), "d"),
                    horizontalalignment="center",
                    color=(
                        "white"
                        if matrix[i, j] < np.sum(matrix) / (len(display_labels) + 1)
                        else "black"
                    ),
                    fontsize="large",
                )
        ax.set_title(f"Confusion Matrix for T5-Sentinel (Variant {variant_level})")
        fig.tight_layout()
        fig.savefig(Path(f"./result/t5_sentinel/confusion_variant{variant_level}.pdf"))

    return implement


def plot_full_confusion_mat_compare():
    fig: plt.Figure = plt.figure(dpi=200, figsize=(8, 8))
    for idx, variant in enumerate(
        [
            "test.variant1.jsonl",
            "test.variant2.jsonl",
            "test.variant3.jsonl",
            "test.variant4.jsonl",
        ]
    ):
        predictions = T5_Full.evaluate_predictions(
            [
                Path("data", "split", "Human", variant),
                Path("data", "split", "GPT-4", variant),
                Path("data", "split", "Claude-Instant-v1", variant),
                Path("data", "split", "Gemini-Pro", variant),
            ]
        )
        display_labels = ["Human", "GPT-4", "Claude 3.5 Haiku", "Gemini 2.0 Flash"]

        matrix = calculate_confusion_matrix(
            predictions, ["Human", "ChatGPT", "Claude", "Gemini"]
        )
        ax: plt.Axes = fig.add_subplot(2, 2, idx + 1)
        ax.imshow(
            matrix,
            cmap=sns.color_palette("crest_r", as_cmap=True),
            interpolation="nearest",
        )
        ax.set_xticks([_ for _ in range(len(display_labels))], display_labels)
        ax.set_yticks([_ for _ in range(len(display_labels))], display_labels)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        for i in range(len(display_labels)):
            for j in range(len(display_labels)):
                ax.text(
                    j,
                    i,
                    format(int(matrix[i, j]), "d"),
                    horizontalalignment="center",
                    color=(
                        "white"
                        if matrix[i, j] < np.sum(matrix) / (len(display_labels) + 1)
                        else "black"
                    ),
                )
    fig.tight_layout()
    fig.savefig(Path(f"./result/t5_sentinel/confusion_compare_variants.pdf"))


if __name__ == "__main__":
    TASKS = [
        plot_full_confusion_mat,
        plot_full_confusion_mat_abalation("test.variant1.jsonl", 1),
        plot_full_confusion_mat_abalation("test.variant2.jsonl", 2),
        plot_full_confusion_mat_abalation("test.variant3.jsonl", 3),
        plot_full_confusion_mat_abalation("test.variant4.jsonl", 4),
        # plot_full_confusion_mat_compare
    ]

    for task in TASKS:
        print("Executing task: ", task.__name__)
        task()
