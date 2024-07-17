import csv
import pathlib
import datetime
from tensorflow import keras


def logger(
    history: keras.callbacks.History, file_name: str = "log"
) -> None:
    home_dir = pathlib.Path(__file__).parent.parent.parent.as_posix()
    # CSV file path
    time = str(datetime.datetime.now().strftime(format="%d.%m.%y_%H.%M"))
    csv_file_path = f"{home_dir}/training_logs/{file_name}_{time}.csv"

    # Write the history data to a CSV file
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(["epoch"] + list(history.history.keys()))

        # Write the data
        for i in range(len(history.epoch)):
            row = [i + 1] + [history.history[key][i] for key in history.history]
            writer.writerow(row)
