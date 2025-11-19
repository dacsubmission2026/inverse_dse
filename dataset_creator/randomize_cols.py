import pandas as pd
import numpy as np
import argparse

# ================= ARGUMENTS =================
parser = argparse.ArgumentParser(description="Random architecture sweep generator")
parser.add_argument("-safe", action="store_true", help="Generate only safe configurations")
args = parser.parse_args()

NUM_SAMPLES = int(input("Enter number of cases: "))
SAFE_MODE = args.safe

# ================= PARAMETER SPACE =================
param_space = {
    "num_cores": [1, 2, 4, 8, 16, 32],
    "cpu_clock_GHz": np.round(np.linspace(0.5, 3.0, 25), 2),
    "l1i_kb": [16, 32, 64, 128, 256],
    "l1d_kb": [16, 32, 64, 128, 256],
    "l1_assoc": [1, 2, 4, 8],
    "l2_kb": [128, 256, 512, 1024, 2048],
    "l2_assoc": [2, 4, 8, 16],
    "fetchWidth": [4, 8, 10, 12],
    "decodeWidth": [4, 8, 10, 12],
    "renameWidth": [4, 8, 10, 12],
    "dispatchWidth": [4, 8, 10, 12],
    "issueWidth": [4, 8, 10, 12],
    "commitWidth": [4, 8, 10, 12],
    "wbWidth": [6, 8, 10, 12],
    "numROBEntries": [32, 64, 128, 192, 256],
    "numIQEntries": [16, 32, 64, 96, 128],
    "numPhysIntRegs": [64, 128, 256, 512],
    "numPhysFloatRegs": [64, 128, 256, 512],
    "LQEntries": [8, 16, 32, 64],
    "SQEntries": [8, 16, 32, 64],
    "branch_predictor": [
        "BiModeBP",
        "LocalBP",
        "TAGE",
        "TAGE_SC_L_64KB",
        "MultiperspectivePerceptron64KB",
        "TournamentBP"
    ],
}

workloads = ["balanced", "ilp", "matrixmul", "mcfmini"]
weights = [0.5, 0.2, 0.15, 0.15]

# ================= SANITY CHECK FUNCTION =================
def is_safe_config(cfg):
    # CPU widths
    if cfg['fetchWidth'] > cfg['decodeWidth'] * 2:
        return False
    if cfg['decodeWidth'] > cfg['dispatchWidth'] * 2:
        return False
    if cfg['dispatchWidth'] > cfg['issueWidth'] * 2:
        return False
    if cfg['issueWidth'] > cfg['commitWidth'] * 2:
        return False
    if cfg['wbWidth'] > cfg['commitWidth']:
        return False
    
    # ROB / IQ / LQ / SQ
    if cfg['numROBEntries'] < cfg['commitWidth'] * 2:
        return False
    if cfg['numIQEntries'] < cfg['issueWidth'] * 2:
        return False
    if cfg['LQEntries'] < cfg['commitWidth']:
        return False
    if cfg['SQEntries'] < cfg['commitWidth']:
        return False
    
    # Physical registers
    if cfg['numPhysIntRegs'] < cfg['numROBEntries'] // 2:
        return False
    if cfg['numPhysFloatRegs'] < cfg['numROBEntries'] // 2:
        return False
    
    # Caches
    if cfg['l1i_kb'] < 16 or cfg['l1d_kb'] < 16:
        return False
    if cfg['l2_kb'] < cfg['l1i_kb'] + cfg['l1d_kb']:
        return False
    
    return True

# ================= DATA GENERATION =================
data = []
i = 0
while len(data) < NUM_SAMPLES:
    sample = {"exp_id": i}
    for param, choices in param_space.items():
        choices = np.array(choices)
        value = np.random.choice(choices)
        sample[param] = float(value) if param == "cpu_clock_GHz" else value

    sample["workload"] = np.random.choice(workloads, p=weights)
    
    if SAFE_MODE:
        if not is_safe_config(sample):
            continue
    
    data.append(sample)
    i += 1

df = pd.DataFrame(data)

output_file = "arch_sweep_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Generated {NUM_SAMPLES} {'safe ' if SAFE_MODE else ''}samples and saved to {output_file}")
print(df.head())
