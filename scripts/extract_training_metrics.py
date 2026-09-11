import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("~/ML/robot-learning-project1").expanduser()

LOG_PATH = (
    PROJECT_ROOT
    / "results/logs/act_bowl_plate_baseline.log"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/act_baseline3/training_metrics.csv"
)


def parse_number(text):
    """Parse LeRobot numbers such as 100, 1K, 1.5K."""
    multipliers = {
        "K": 1_000,
        "M": 1_000_000,
    }

    if text[-1:] in multipliers:
        return float(text[:-1]) * multipliers[text[-1]]

    return float(text)


patterns = {
    "step": r"\bstep:([0-9.]+[KM]?)",
    "loss": r"\bloss:([0-9.eE+-]+)",
    "l1_loss": r"\bl1_loss:([0-9.eE+-]+)",
    "kld_loss": r"\bkld_loss:([0-9.eE+-]+)",
    "grad_norm": r"\bgrdn:([0-9.eE+-]+)",
    "lr": r"\blr:([0-9.eE+-]+)",
    "mem_gb": r"\bmem_gb:([0-9.eE+-]+)",
}

rows = []

with LOG_PATH.open(errors="replace") as f:
    for line in f:
        # Training metric lines always contain both step and loss
        if "step:" not in line or "loss:" not in line:
            continue

        row = {}

        for name, pattern in patterns.items():
            match = re.search(pattern, line)

            if match:
                value = match.group(1)

                if name == "step":
                    value = int(parse_number(value))
                else:
                    value = float(value)

                row[name] = value

        if "step" in row and "loss" in row:
            rows.append(row)


df = pd.DataFrame(rows)

# Avoid accidental duplicated log lines from resumed / appended runs
df = (
    df.drop_duplicates(subset="step", keep="last")
    .sort_values("step")
    .reset_index(drop=True)
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(df)
print("\nSaved:", OUTPUT_PATH)