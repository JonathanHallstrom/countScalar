import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd
import numpy as np


def sizeof_fmt(x, pos):
    if x < 0:
        return ""
    for x_unit in ["bytes", "kB", "MB", "GB", "TB"]:
        if x < 1024:
            return "%3.0f %s" % (x, x_unit)
        x /= 1024


def time_fmt(x, pos):
    if x < 0:
        return ""
    for x_unit in ["ns", "us", "ms", "s"]:
        if x < 1000:
            return "%3.1f %s" % (x, x_unit)
        x /= 1000

def cycle_fmt(x, pos):
    return "%3.1f" % x


def main(name):
    file = open(name.split("_")[0] + ".csv", "r")
    lines = file.readlines()

    data = []

    labels = lines[0].split(",")
    for i in range(1, len(lines)):
        data_points = [float(x) for x in lines[i].split(",")]
        data.append(data_points)

    file.close()

    df = pd.DataFrame(data, columns=labels)

    df_to_plot = df

    plt.figure(figsize=(14, 10))
    inf = float("inf")
    lo, hi = inf, -inf
    for label in labels[1:]:
        dashes = []
        smoothed_data = df_to_plot[label].rolling(50).median()
        filter_nan = lambda a: a[~np.isnan(a)]
        lo = min(lo, min(filter_nan(smoothed_data)))
        hi = max(hi, max(filter_nan(smoothed_data)))

        plt.plot(
            df_to_plot["size"],
            smoothed_data,
            label=label,
            dashes=dashes,
        )

    plt.gca().set_yscale("log")

    plt.gca().set_xscale("log")
    plt.gca().xaxis.set_major_formatter(tkr.FuncFormatter(sizeof_fmt))
    plt.gca().yaxis.set_major_locator(
        tkr.LogLocator(base=10.0, subs=range(0, 10, 1), numticks=10)
    )
    # plt.gca().yaxis.set_minor_locator(
    #     tkr.LogLocator(base=10.0, subs=range(0, 10, 1), numticks=10)
    # )
    plt.gca().yaxis.set_major_formatter(tkr.FuncFormatter(cycle_fmt))
    plt.gca().yaxis.set_minor_formatter(tkr.FuncFormatter(lambda *a: ""))
    plt.gca().grid(True, which="both", linestyle="--", linewidth=1)
    plt.xlabel("Size (bytes)")
    plt.ylabel("Cycles")

    plt.title(f"countScalar ({name})")

    plt.legend()

    plt.savefig(name + ".png")
    plt.show()
    plt.close()


main("u8")
main("u16")
main("u32")
main("u64")
