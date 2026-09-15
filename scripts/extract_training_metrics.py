import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("~/ML/robot-learning-project1").expanduser()

LOG_PATH = (
    PROJECT_ROOT
    / "results/logs/diffusion_bowl_plate_baseline2.log"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/diffusion_baseline/training_metrics2.csv"
)

LOG_FREQ = 50
steps = 5000


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
    "nominal_step": r"\bstep:([0-9.]+[KM]?)",
    "smpl": r"\bsmpl:([0-9.eE+-]+)",
    "ep": r"\bep:([0-9.eE+-]+)",
    "epoch": r"\bepoch:([0-9.eE+-]+)",
    "loss": r"\bloss:([0-9.eE+-]+)",
    "l1_loss": r"\bl1_loss:([0-9.eE+-]+)",
    "kld_loss": r"\bkld_loss:([0-9.eE+-]+)",
    "grad_norm": r"\bgrdn:([0-9.eE+-]+)",
    "grdn": r"\bgrdn:([0-9.eE+-]+)",
    "lr": r"\blr:([0-9.eE+-]+)",
    "data_s": r"\bdata_s:([0-9.eE+-]+)",
    "prep_s": r"\bprep_s:([0-9.eE+-]+)",
    "updt_s": r"\bupdt_s:([0-9.eE+-]+)",
    "step_s": r"\bstep_s:([0-9.eE+-]+)",
    "smp/s": r"\bsmp/:([0-9.eE+-]+)",
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

                if name == "nominal_step":
                    value = int(parse_number(value))
                else:
                    value = float(value)

                row[name] = value

        if "nominal_step" in row and "loss" in row:
            rows.append(row)


df = pd.DataFrame(rows)

assert abs(len(df)*LOG_FREQ - steps) <1e-5
# Treat the duplicates as the rounded steps with the same nominal_step (eg., 1500 -> 2K)
df.insert(
    0,
    "step",
    [(i + 1) * LOG_FREQ for i in range(len(df))]
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print(df)
print("\nSaved:", OUTPUT_PATH)