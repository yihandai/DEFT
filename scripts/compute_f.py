import json
import re
from collections import defaultdict
import os
import numpy as np


def parse_log_file(log_path):
    """
    Parse the log file to extract insertion and deletion samples for each instruction ID and mask percentage.

    Returns a dictionary with structure:
    {
        "instr_id": {
            "mask": {
                0.25: {"Insertion": [0., 1., ...], "Deletion": [0., 1., ...]},
                0.5: {"Insertion": [0., 1., ...], "Deletion": [0., 1., ...]},
                0.75: {"Insertion": [0., 1., ...], "Deletion": [0., 1., ...]}
            }
        }
    }
    """
    results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    current_instr_id = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # Check if line is an instruction ID (pattern: digits_underscore_digits)
            instr_id_match = re.match(r"^(\d+_\d+)$", line)
            if instr_id_match:
                current_instr_id = instr_id_match.group(1)
                continue

            # Skip if we haven't found an instruction ID yet
            if current_instr_id is None:
                continue

            # Match Insertion or Deletion sample lines
            # Pattern: "Insertion sample: 0. Over-all: ... for mask percentage: 0.25"
            # or "Deletion sample: 1. Over-all: ... for mask percentage: 0.5"
            insertion_match = re.search(
                r"Insertion sample: ([01])\.\s+Over-all:.*?for mask percentage: (0\.\d+)",
                line,
            )
            deletion_match = re.search(
                r"Deletion sample: ([01])\.\s+Over-all:.*?for mask percentage: (0\.\d+)",
                line,
            )

            if insertion_match:
                sample_value = float(insertion_match.group(1))
                mask_perc = float(insertion_match.group(2))
                results[current_instr_id]["mask"][mask_perc]["Insertion"].append(
                    sample_value
                )
            elif deletion_match:
                sample_value = float(deletion_match.group(1))
                mask_perc = float(deletion_match.group(2))
                results[current_instr_id]["mask"][mask_perc]["Deletion"].append(
                    sample_value
                )

    # Convert defaultdict to regular dict and ensure mask percentages are sorted
    output = {}
    for instr_id, data in results.items():
        mask_dict = {}
        for mask_perc in sorted(data["mask"].keys()):
            mask_dict[mask_perc] = {
                "Insertion": data["mask"][mask_perc]["Insertion"],
                "Deletion": data["mask"][mask_perc]["Deletion"],
            }
        output[instr_id] = {"mask": mask_dict}

    return output


def main():
    log_path = "scripts/out_smdl.log"
    output_path = "scripts/out_smdl.json"

    print(f"Parsing log file: {log_path}")
    results = parse_log_file(log_path)

    print(f"Found {len(results)} instruction IDs")

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Print summary
    for instr_id, data in list(results.items())[:3]:  # Show first 3 as examples
        print(f"\n{instr_id}:")
        for mask_perc, sample_dict in data["mask"].items():
            insertion_count = len(sample_dict["Insertion"])
            deletion_count = len(sample_dict["Deletion"])
            print(
                f"  {mask_perc}: Insertion={insertion_count}, Deletion={deletion_count}"
            )


def compute_f(output_path, causal_metric_dir):
    with open(output_path, "r") as f:
        results = json.load(f)
    for instr_id, data in results.items():
        for mask_perc, sample_dict in data["mask"].items():
            pass

    consistency_score = []
    importance_score = []
    for instr_id in os.listdir(
        os.path.join(causal_metric_dir, "consistency_importance_score")
    ):
        for t in os.listdir(
            os.path.join(causal_metric_dir, "consistency_importance_score", instr_id)
        ):
            for mask_perc in os.listdir(
                os.path.join(
                    causal_metric_dir,
                    "consistency_importance_score",
                    instr_id,
                    str(t),
                )
            ):
                score = np.load(
                    os.path.join(
                        causal_metric_dir,
                        "consistency_importance_score",
                        instr_id,
                        str(t),
                        str(mask_perc),
                        "score.npy",
                    )
                )
                consistency_score.append(score[0])
                importance_score.append(score[1])
    return self.muFidelity(consistency_score, importance_score)


if __name__ == "__main__":
    main()
