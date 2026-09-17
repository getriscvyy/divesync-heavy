import json
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Set global matplotlib parameters for academic style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",  # Renders math equations matching Times styling
    }
)


class Plotter:

    def __init__(self, folder_path, custom_title=None):
        self._folder_path = folder_path
        self._custom_title = custom_title
        self._df = pd.read_csv(f"{self._folder_path}/processed.csv")
        self._df.dropna(how="all", inplace=True)

    # format: time_s, depth_filtered_m, depth_setpoint_m, actuator_mm, actuator_setpoint_mm, motor_cmd
    def plot(self):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8.5), dpi=100)
        fig.subplots_adjust(hspace=0.35)

        # Line formatting constants
        LINE_WIDTH = 2.5
        SETPOINT_WIDTH = 2.0

        # Default Matplotlib color palette references
        COLOR_ACTUAL = "C0"    # Default Matplotlib blue
        COLOR_SETPOINT = "C1"  # Default Matplotlib orange

        # Font size constants
        TITLE_FONT_SIZE = 18
        SUBTITLE_FONT_SIZE = 14
        LABEL_FONT_SIZE = 14
        TICK_FONT_SIZE = 12
        LEGEND_FONT_SIZE = 12

        # Use the custom title if provided, otherwise fallback to default string formatting
        title_text = (
            self._custom_title
            if self._custom_title
            else f"EXPERIMENT {self._folder_path[5:]} RESULTS"
        )
        fig.suptitle(title_text, fontsize=TITLE_FONT_SIZE, fontweight="bold")

        time = self._df["time_s"]

        # Depth plot (plot 1)
        depth = self._df["depth_filtered_m"]
        axes[0].plot(
            time, depth, label="Depth (m)", color=COLOR_ACTUAL, lw=LINE_WIDTH
        )
        if self._df["depth_setpoint_m"].notna().any():
            depth_setpoint = self._df["depth_setpoint_m"]
            axes[0].plot(
                time,
                depth_setpoint,
                label="Depth Setpoint (m)",
                color=COLOR_SETPOINT,
                ls="--",
                lw=SETPOINT_WIDTH,
            )
        axes[0].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].set_title(
            r"Depth (m)",
            loc="center",
            fontsize=SUBTITLE_FONT_SIZE,
        )
        axes[0].tick_params(axis="both", labelsize=TICK_FONT_SIZE)

        # Actuator plot (plot 2)
        actuator = self._df["actuator_mm"]
        axes[1].plot(
            time,
            actuator,
            label="Actuator Position (mm)",
            color=COLOR_ACTUAL,
            lw=LINE_WIDTH,
        )
        if self._df["actuator_setpoint_mm"].notna().any():
            actuator_setpoint = self._df["actuator_setpoint_mm"]
            axes[1].plot(
                time,
                actuator_setpoint,
                label="Actuator Setpoint (mm)",
                color=COLOR_SETPOINT,
                ls="--",
                lw=SETPOINT_WIDTH,
            )
        axes[1].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].set_title(
            r"Piston Position (mm)",
            loc="center",
            fontsize=SUBTITLE_FONT_SIZE,
        )
        axes[1].tick_params(axis="both", labelsize=TICK_FONT_SIZE)

        # PWM plot (plot 3)
        pwm = self._df["motor_cmd"]
        axes[2].plot(
            time, pwm, label="Motor Voltage (V)", color=COLOR_ACTUAL, lw=LINE_WIDTH
        )
        axes[2].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].set_title(
            "Motor Voltage (V)",
            loc="center",
            fontsize=SUBTITLE_FONT_SIZE,
        )
        axes[2].tick_params(axis="both", labelsize=TICK_FONT_SIZE)

        axes[2].set_xlabel(
            "Time (s)", fontsize=LABEL_FONT_SIZE, fontweight="bold"
        )

        plt.show()


if __name__ == "__main__":
    from pathlib import Path

    # Map target path to relative directory
    data_dir = Path("data")

    if not data_dir.exists() or not data_dir.is_dir():
        print("Error: The 'data' directory does not exist.\n")
        sys.exit(1)

    print("=== DIVESYNC HEAVY - DATA PLOTTING AND REPLAY INTERFACE ===\n")

    def get_notes(folder):
        metadata_path = Path(folder) / "metadata.json"
        if not metadata_path.exists():
            return ""
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("notes", "")
        except (json.JSONDecodeError, OSError):
            return ""

    def choose_subfolder(folder):
        """Prompt the user to pick a subfolder of `folder`, showing notes where
        available. Returns the chosen Path.
        """
        subfolders = sorted([f for f in folder.iterdir() if f.is_dir()])
        if not subfolders:
            return None

        for idx, sub in enumerate(subfolders):
            notes = get_notes(sub)
            if notes:
                print(f"[{idx}] {sub.name} - {notes}")
            else:
                print(f"[{idx}] {sub.name}")
        print()

        while True:
            try:
                idx = int(input("Select a folder (index): ").strip())
                return subfolders[idx]
            except ValueError:
                print("Error: please enter a valid number\n")
            except IndexError:
                if len(subfolders) == 1:
                    print("Error: enter 0")
                else:
                    print(
                        f"Error: enter a number between 0 and {len(subfolders) - 1}\n"
                    )

    print("Available experiments:\n")
    current_folder = data_dir

    while True:
        chosen = choose_subfolder(current_folder)
        if chosen is None:
            print("No folders available. Exiting plotter.\n")
            sys.exit(0)

        if (chosen / "processed.csv").exists():
            selected_folder_path = str(chosen)
            print(f"\nSelected folder path: {selected_folder_path}\n")
            break

        # No processed.csv directly here -- descend into it and show contents
        print(
            f"\n'{chosen.name}' has no processed.csv directly -- browsing its"
            " contents:\n"
        )
        current_folder = chosen

    # Prompt user for custom figure title
    user_title = input(
        "Enter a title for this run (press Enter for default): "
    ).strip()

    print("\nPlotting...\n")

    # Replay execution
    try:
        plotter = Plotter(
            folder_path=selected_folder_path, custom_title=user_title
        )
        plotter.plot()
    except FileNotFoundError:
        print(
            f"Error: 'processed.csv' not found inside {selected_folder_path}\n"
        )
