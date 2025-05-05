import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
file_paths = {
    "7": {
        "modeled": "flight_modeled7_before.csv",
        "interpolation": "flight_modeled7.csv"
    },
    "8": {
        "modeled": "flight_modeled8_before.csv",
        "interpolation": "flight_modeled8.csv"
    }
}

# Process each flight
for flight_id, paths in file_paths.items():
    try:
        print(f"Processing flight {flight_id}...")

        # Load only required columns
        df_modeled = pd.read_csv(paths["modeled"], usecols=["timepoint", "CH4"])
        df_interp = pd.read_csv(paths["interpolation"], usecols=["timepoint", "CH4"])

        # Convert time to datetime
        df_modeled["Time"] = pd.to_datetime(df_modeled["timepoint"])
        df_interp["Time"] = pd.to_datetime(df_interp["timepoint"])
        df_modeled = df_modeled.set_index("Time").resample('1s').interpolate().reset_index()
        df_interp = df_interp.set_index("Time").resample('1s').interpolate().reset_index()
        # Calculate CH4 difference
        ch4_diff = df_modeled["CH4"] - df_interp["CH4"]
        ch4_modeled = df_modeled["CH4"]
        ch4_interp = df_interp["CH4"]

        # Plot and save
        plt.figure(figsize=(10, 5))
        # plt.plot(df_modeled["Time"], ch4_modeled, label="CH4 without interpolation", linewidth=1)
        # plt.plot(df_modeled["Time"], ch4_interp, label="CH4 with interpolation", linewidth=1)
        plt.plot(df_modeled["Time"], ch4_diff, label="CH4 diff", linewidth=1)
        plt.title(f"Flight {flight_id} - CH4 (Modeled - Interpolation)")
        plt.xlabel("Time")
        plt.ylabel("CH4")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        filename = f"flight_{flight_id}_CH4_difference.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to: {filename}")

    except Exception as e:
        print(f"Error processing flight {flight_id}: {e}")