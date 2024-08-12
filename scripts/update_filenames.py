import os
import re

directory = "./assets/results/by_series"

for filename in os.listdir(directory):
    if re.match(r"^[^_]+_[^_]+_[^_]+_.*_classical\.csv$", filename):
        match = re.match(r"^([^_]+_[^_]+_[^_]+_)(.*)(_classical\.csv)$", filename)
        if match:
            prefix, middle, suffix = match.groups()

            middle_transformed = (
                middle.replace(".", "")
                .replace("/", "-")
                .replace("\\", "-")
                .replace("_", "-")
                .replace(",", "")
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
            )

            new_filename = f"{prefix}{middle_transformed}{suffix}"

            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)

            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
