import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

def plot_dataframe(df, png_path, title):
    cols = [c for c in df.columns if c != "hh_id"]
    n = len(cols)

    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    fig.suptitle(title, fontsize=18, y=1.02)

    for ax, col in zip(axes, cols):
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].plot(kind="hist", bins=30, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
        else:
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

        ax.set_title(col)

    for i in range(len(cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    rural_path = os.path.join(input_dir, "rural.csv")
    urban_path = os.path.join(input_dir, "urban.csv")

    if not (os.path.isfile(rural_path) and os.path.isfile(urban_path)):
        print("Error: rural.csv and/or urban.csv not found.")
        sys.exit(1)

    rural_df = pd.read_csv(rural_path)
    urban_df = pd.read_csv(urban_path)

    rural_png = os.path.join(output_dir, "rural_plots.png")
    urban_png = os.path.join(output_dir, "urban_plots.png")

    plot_dataframe(rural_df, rural_png, "Rural")
    plot_dataframe(urban_df, urban_png, "Urban")

    print("PNG plots created successfully.")


if __name__ == "__main__":
    main()

