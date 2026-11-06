"""Create visualizations for FraudShield cost sensitivity results.

Input:
    results/cost_sensitivity.csv

Outputs:
    results/figures/cost_sensitivity_heatmaps.png
    results/figures/reviewer_accuracy_comparison.png
    results/figures/cost_vs_silent_miss.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("results/cost_sensitivity.csv")
OUTPUT_DIR = Path("results/figures")


def load_results() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT}. Run python run_cost_sensitivity.py first."
        )

    df = pd.read_csv(INPUT)
    required = {
        "false_block_cost",
        "review_cost",
        "reviewer_accuracy",
        "cost_savings_pct",
        "silent_miss_rate",
        "defer_rate",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    return df


def plot_heatmaps(df: pd.DataFrame) -> None:
    accuracies = sorted(df["reviewer_accuracy"].unique())
    fig, axes = plt.subplots(
        1, len(accuracies),
        figsize=(5 * len(accuracies), 4.5),
        squeeze=False,
    )

    for ax, accuracy in zip(axes[0], accuracies):
        subset = df[df["reviewer_accuracy"] == accuracy]
        pivot = subset.pivot(
            index="false_block_cost",
            columns="review_cost",
            values="cost_savings_pct",
        ).sort_index(ascending=True)

        image = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"Reviewer accuracy: {accuracy:.0%}")
        ax.set_xlabel("Review cost")
        ax.set_ylabel("False-block cost")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{x:g}" for x in pivot.index])

        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iloc[row, col]
                ax.text(col, row, f"{value:.1f}%", ha="center", va="center")

        fig.colorbar(image, ax=ax, label="Modelled cost savings (%)")

    fig.suptitle("FraudShield cost sensitivity across assumptions")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cost_sensitivity_heatmaps.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_accuracy_comparison(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["false_block_cost", "reviewer_accuracy"], as_index=False)
        ["cost_savings_pct"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for false_block_cost, subset in grouped.groupby("false_block_cost"):
        subset = subset.sort_values("reviewer_accuracy")
        ax.plot(
            subset["reviewer_accuracy"] * 100,
            subset["cost_savings_pct"],
            marker="o",
            label=f"False-block cost = {false_block_cost:g}",
        )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("Effect of reviewer accuracy on modelled cost savings")
    ax.set_xlabel("Reviewer accuracy (%)")
    ax.set_ylabel("Mean modelled cost savings (%)")
    ax.legend(title="Assumption")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "reviewer_accuracy_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cost_vs_miss_rate(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for accuracy, subset in df.groupby("reviewer_accuracy"):
        ax.scatter(
            subset["silent_miss_rate"] * 100,
            subset["cost_savings_pct"],
            label=f"Reviewer accuracy = {accuracy:.0%}",
            alpha=0.8,
        )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("Modelled cost savings versus silent fraud miss rate")
    ax.set_xlabel("Silent miss rate (%)")
    ax.set_ylabel("Modelled cost savings (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cost_vs_silent_miss.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    plot_heatmaps(df)
    plot_accuracy_comparison(df)
    plot_cost_vs_miss_rate(df)

    print("Created three cost sensitivity figures:")
    for filename in [
        "cost_sensitivity_heatmaps.png",
        "reviewer_accuracy_comparison.png",
        "cost_vs_silent_miss.png",
    ]:
        print(f" - {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
