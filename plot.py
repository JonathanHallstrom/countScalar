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


def handle_type(element_type, mode):
    file = open(mode + "_" + element_type + ".csv", "r")
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
    for label in labels[1:]:
        dashes = []
        smoothed_data = df_to_plot[label].rolling(50).median()

        plt.plot(
            df_to_plot["size"],
            smoothed_data,
            label=label,
            dashes=dashes,
        )

    plt.gca().set_yscale("log")

    plt.gca().set_xscale("log")
    plt.gca().xaxis.set_major_locator(
        tkr.LogLocator(base=10.0, subs=range(0, 11, 3), numticks=10)
    )
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
    plt.ylabel("Cycles" if mode == "cycles" else "Bytes per cycle")

    plt.title(f"countScalar ({element_type})")

    plt.legend()

    plt.savefig(mode + "_" + element_type + ".png")
    plt.show()
    plt.close()


handle_type("u8", "cycles")
handle_type("u8", "bytes_per_cycle")
handle_type("u16", "cycles")
handle_type("u16", "bytes_per_cycle")
handle_type("u32", "cycles")
handle_type("u32", "bytes_per_cycle")
handle_type("u64", "cycles")
handle_type("u64", "bytes_per_cycle")
