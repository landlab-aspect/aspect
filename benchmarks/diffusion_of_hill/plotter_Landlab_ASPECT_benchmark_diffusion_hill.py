

"""
===============================================================================
Surface Topography Benchmark Plotter
===============================================================================

This script reproduces the benchmark plotting workflow used for the
Landlab-ASPECT diffusion hill benchmark.

It generates:

1. Surface topography comparison:
      - Landlab-ASPECT solution
      - Analytical solution

2. Percentage error plot

Input files expected:

    topography.00060
    topography.00120
    topography.00180

Each file must contain at least four columns:

    column 0 : x coordinate
    column 2 : ASPECT topography
    column 3 : analytical topography

Output:

    result_benchmark_ASPECT/
        surface_topo_benchmark_ASPECT_<run_folder>.png
        surface_topo_error_benchmark_ASPECT_<run_folder>.png

===============================================================================
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_FOLDER = "output-1_sine_zero_flux_landlab"

# Plot ranges

X_RANGE = [0.0, 1.0]

Y_RANGE_TOPO = [0.0, 0.1]

Y_RANGE_ERROR = [0.0, 2.0]


# =============================================================================
# INPUT FILES
# =============================================================================

TIMES = [
    ("00060", "60", "#1f77b4"),
    ("00120", "120", "#d62728"),
    ("00180", "180", "#2ca02c"),
]


# =============================================================================
# FIGURE STYLE
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.linewidth": 2,
    }
)


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_topography_file(filename):
    """
    Read benchmark topography file.
    """

    data = np.loadtxt(filename)

    x = data[:, 0]
    aspect = data[:, 2]
    analytical = data[:, 3]

    return x, aspect, analytical


def compute_mad(aspect, analytical):
    """
    Mean Absolute Difference.
    """

    return np.mean(np.abs(aspect - analytical))


def compute_percent_error(aspect, analytical):
    """
    Percentage error.

    Protect against divide-by-zero.
    """

    return np.where(
        np.abs(analytical) > 1.0e-12,
        np.abs((analytical - aspect) / analytical) * 100.0,
        0.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    output_dir = Path(RUN_FOLDER)

    result_dir = output_dir / "result_benchmark_ASPECT"

    result_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # TOPOGRAPHY PLOT
    # =========================================================================

    fig, ax = plt.subplots(figsize=(18, 10))

    mad_values = []

    for step, time_label, color in TIMES:

        file_path = output_dir / f"topography.{step}"

        if not file_path.exists():
            print(f"WARNING: Missing file: {file_path}")
            continue

        x, aspect, analytical = load_topography_file(file_path)

        mad = compute_mad(aspect, analytical)

        mad_values.append((time_label, mad, color))

        # ---------------------------------------------------------------------
        # ASPECT solution
        # ---------------------------------------------------------------------

        ax.plot(
            x,
            aspect,
            marker="o",
            markersize=6,
            linewidth=3,
            color=color,
            label=f"ASPECT t={time_label}",
        )

        # ---------------------------------------------------------------------
        # Analytical solution
        # ---------------------------------------------------------------------

        ax.plot(
            x,
            analytical,
            linestyle="--",
            linewidth=4,
            color=color,
            label=f"Analytical t={time_label}",
        )

    # -------------------------------------------------------------------------
    # Styling
    # -------------------------------------------------------------------------

    ax.set_xlim(X_RANGE)

    ax.set_ylim(Y_RANGE_TOPO)

    ax.set_xlabel("X [-]", fontsize=22)

    ax.set_ylabel("Topography [m]", fontsize=22)

    ax.set_title(
        f"Surface Topography Benchmark\n{RUN_FOLDER}",
        fontsize=26,
    )

    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper right",
        framealpha=1.0,
        fontsize=16,
    )

    # -------------------------------------------------------------------------
    # MAD annotations
    # -------------------------------------------------------------------------

    ypos = 0.92

    for time_label, mad, color in mad_values:

        ax.text(
            0.02,
            ypos,
            f"MAD t={time_label} = {mad:.3e} m",
            transform=ax.transAxes,
            color=color,
            fontsize=16,
        )

        ypos -= 0.05

    topo_png = (
        result_dir
        / f"surface_topo_benchmark_ASPECT_{RUN_FOLDER}.png"
    )

    fig.savefig(
        topo_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # ERROR PLOT
    # =========================================================================

    fig, ax = plt.subplots(figsize=(18, 10))

    for step, time_label, color in TIMES:

        file_path = output_dir / f"topography.{step}"

        if not file_path.exists():
            continue

        x, aspect, analytical = load_topography_file(file_path)

        error = compute_percent_error(
            aspect,
            analytical,
        )

        ax.plot(
            x,
            error,
            marker="o",
            markersize=6,
            linewidth=3,
            color=color,
            label=f"t={time_label}",
        )

    ax.set_xlim(X_RANGE)

    ax.set_ylim(Y_RANGE_ERROR)

    ax.set_xlabel("X [-]", fontsize=22)

    ax.set_ylabel("Topography Error [%]", fontsize=22)

    ax.set_title(
        f"Surface Topography Error Benchmark\n{RUN_FOLDER}",
        fontsize=26,
    )

    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper center",
        framealpha=1.0,
        fontsize=16,
    )

    error_png = (
        result_dir
        / f"surface_topo_error_benchmark_ASPECT_{RUN_FOLDER}.png"
    )

    fig.savefig(
        error_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("============================================================")
    print("Benchmark plots generated successfully")
    print("============================================================")
    print()
    print("Topography plot:")
    print(f"  {topo_png}")
    print()
    print("Error plot:")
    print(f"  {error_png}")
    print()


if __name__ == "__main__":
    main()

