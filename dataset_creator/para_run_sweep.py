import os
import pandas as pd
import traceback
import re
import subprocess
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= CONFIGURATION =================
HOME = os.path.expanduser("~")
csv_path = "arch_sweep_dataset.csv"

GEM5_BIN = os.path.join(HOME, "gem5", "build", "ALL", "gem5.fast")
SE_SCRIPT = os.path.join(HOME, "gem5", "configs", "deprecated", "example", "se.py")

WORKLOADS = {
    "balanced": os.path.join(HOME, "gem5/benchmarks/bin/balanced"),
    "ilp": os.path.join(HOME, "gem5/benchmarks/bin/ilp"),
    "matrixmul": os.path.join(HOME, "gem5/benchmarks/bin/matrixmul"),
    "mcfmini": os.path.join(HOME, "gem5/benchmarks/bin/mcfmini"),
}

OUTDIR = os.path.join(os.getcwd(), "output")
MCPAT_BIN = os.path.join(HOME, "mcpat", "mcpat")
PARSER_DIR = os.path.join(HOME, "Gem5McPATParser")

# ================= TEST MODE ARGUMENT =================
parser = argparse.ArgumentParser(description="Run gem5 + McPAT sweep")
parser.add_argument('-test', action='store_true', help='Run only the first case (test mode)')
args = parser.parse_args()
TEST_MODE = args.test

# ================= PARAMETERS =================
o3_params = [
    "fetchWidth", "decodeWidth", "renameWidth", "dispatchWidth",
    "issueWidth", "commitWidth", "wbWidth",
    "numROBEntries", "numIQEntries", "numPhysIntRegs", "numPhysFloatRegs",
    "LQEntries", "SQEntries",
]

# ================= MC PAT & Stats Parsing =================
def parse_mcpat_output(file_path: str, row: pd.Series) -> dict:
    bp_name = row['branch_predictor']

    bp_metrics_relative = {
        "LocalBP": {"power": 0.48, "area": 0.5},
        "BiModeBP": {"power": 0.6, "area": 0.7},
        "TournamentBP": {"power": 1.0, "area": 1.0},
        "TAGE": {"power": 1.28, "area": 1.33},
        "TAGE_SC_L_64KB": {"power": 1.44, "area": 1.5},
        "MultiperspectivePerceptron64KB": {"power": 1.8, "area": 2.0}
    }

    tournament_bp_metrics_all = {
        4: {"Area":3.88109,"Peak Dynamic":0.683557,"Subthreshold Leakage":0.11562,"Gate Leakage":0.00448018,"Runtime Dynamic":0.0211189,"Peak Power":0,"Total Leakage":0},
        8: {"Area":13.4925,"Peak Dynamic":2.30172,"Subthreshold Leakage":0.186329,"Gate Leakage":0.00693452,"Runtime Dynamic":0.0352906,"Peak Power":0,"Total Leakage":0},
        10: {"Area": 21.193, "Peak Dynamic": 3.5743, "Subthreshold Leakage": 0.22436, "Gate Leakage": 0.008364, "Runtime Dynamic": 0.04234, "Peak Power": 0, "Total Leakage": 0 },
        12: {"Area":28.8935,"Peak Dynamic":4.84693,"Subthreshold Leakage":0.262391,"Gate Leakage":0.00979394,"Runtime Dynamic":0.0493835,"Peak Power":0,"Total Leakage":0}
    }

    with open(file_path, "r") as f:
        content = f.read()

    patterns = {
        "full": {
            'Area': re.compile(r'Area\s*=\s*([\d\.]+)\s*mm\^2'),
            'Peak Power': re.compile(r'Peak Power\s*=\s*([\d\.]+)\s*W'),
            'Total Leakage': re.compile(r'Total Leakage\s*=\s*([\d\.]+)\s*W'),
            'Peak Dynamic': re.compile(r'Peak Dynamic\s*=\s*([\d\.]+)\s*W'),
            'Subthreshold Leakage': re.compile(r'Subthreshold Leakage\s*=\s*([\d\.]+)\s*W'),
            'Gate Leakage': re.compile(r'Gate Leakage\s*=\s*([\d\.]+)\s*W'),
            'Runtime Dynamic': re.compile(r'Runtime Dynamic\s*=\s*([\d\.]+)\s*W'),
        },
        "bp": {
            'Area': re.compile(r'Branch Predictor:\s*Area\s*=\s*([\d\.]+)\s*mm\^2', re.DOTALL),
            'Peak Dynamic': re.compile(r'Branch Predictor:.*Peak Dynamic\s*=\s*([\d\.]+)\s*W', re.DOTALL),
            'Subthreshold Leakage': re.compile(r'Branch Predictor:.*Subthreshold Leakage\s*=\s*([\d\.]+)\s*W', re.DOTALL),
            'Gate Leakage': re.compile(r'Branch Predictor:.*Gate Leakage\s*=\s*([\d\.]+)\s*W', re.DOTALL),
            'Runtime Dynamic': re.compile(r'Branch Predictor:.*Runtime Dynamic\s*=\s*([\d\.]+)\s*W', re.DOTALL),
        }
    }

    total_metrics = {}
    bp_metrics = {'Peak Power' : 0, 'Total Leakage': 0}

    for key, pattern in patterns['full'].items():
        match = pattern.search(content)
        total_metrics[key] = float(match.group(1)) if match else None

    for key, pattern in patterns['bp'].items():
        match = pattern.search(content)
        bp_metrics[key] = float(match.group(1)) if match else 0.0

    final_metrics = {}
    for key in total_metrics:
        if total_metrics[key] is not None:
            scale = bp_metrics_relative[bp_name]["area"] if key == "Area" else bp_metrics_relative[bp_name]["power"]
            tournament_value = tournament_bp_metrics_all[row['decodeWidth']].get(key, 0.0)
            final_metrics[key] = (total_metrics[key] - bp_metrics.get(key, 0.0)) + tournament_value * scale
        else:
            final_metrics[key] = None

    data = dict(row)
    data.update(final_metrics)
    return data

def parse_stats_file(stats_file: str) -> dict:
    patterns = {
        'ipc': re.compile(r'system\.cpu\.ipc\s+([\d\.Ee+-]+)'),
        'committed_branches': re.compile(r'system\.cpu\.branchPred\.committed_0::total\s+(\d+)'),
        'mispredicted_branches': re.compile(r'system\.cpu\.branchPred\.mispredicted_0::total\s+(\d+)'),
        'icache_read_accesses': re.compile(r'system\.cpu\.icache\.ReadReq\.accesses::total\s+(\d+)'),
        'icache_read_misses': re.compile(r'system\.cpu\.icache\.ReadReq\.misses::total\s+(\d+)'),
        'dcache_read_accesses': re.compile(r'system\.cpu\.dcache\.ReadReq\.accesses::total\s+(\d+)'),
        'dcache_write_accesses': re.compile(r'system\.cpu\.dcache\.WriteReq\.accesses::total\s+(\d+)'),
        'dcache_read_misses': re.compile(r'system\.cpu\.dcache\.ReadReq\.misses::total\s+(\d+)'),
        'dcache_write_misses': re.compile(r'system\.cpu\.dcache\.WriteReq\.misses::total\s+(\d+)'),
    }

    with open(stats_file, "r") as f:
        content = f.read()

    data = {}
    for key, pattern in patterns.items():
        match = pattern.search(content)
        data[key] = float(match.group(1)) if match else None

    def safe_div(num, denom): return (num / denom) if denom else None

    return {
        'ipc': round(data.get('ipc'), 4) if data.get('ipc') is not None else None,
        'branch_misprediction_rate': round(safe_div(data.get('mispredicted_branches'), data.get('committed_branches')), 4) if data.get('committed_branches') else None,
        'icache_miss_rate': round(safe_div(data.get('icache_read_misses'), data.get('icache_read_accesses')), 4) if data.get('icache_read_accesses') else None,
        'dcache_read_miss_rate': round(safe_div(data.get('dcache_read_misses'), data.get('dcache_read_accesses')), 4) if data.get('dcache_read_accesses') else None,
        'dcache_write_miss_rate': round(safe_div(data.get('dcache_write_misses'), data.get('dcache_write_accesses')), 4) if data.get('dcache_write_accesses') else None,
    }

# ================= GEM5 + MCPAT RUN =================
def run_gem5_mcpat(case_id: int, row: pd.Series):
    outdir_case = os.path.join(OUTDIR, f"case_{case_id}")
    os.makedirs(outdir_case, exist_ok=True)

    stats_file = os.path.join(outdir_case, "stats.txt")
    config_ini = os.path.join(outdir_case, "config.ini")
    config_json = os.path.join(outdir_case, "config.json")
    mcpat_xml = os.path.join(outdir_case, "mcpat.xml")
    mcpat_out = os.path.join(outdir_case, "mcpat_power.txt")

    p_flags = [f'-P "system.cpu[0].{param}={int(row[param])}"'
               for param in o3_params if param in row and pd.notna(row[param])]

    WORKLOAD = WORKLOADS[row["workload"]]

    cmd_lines = [
        f'"{GEM5_BIN}"',
        f'--outdir="{outdir_case}"',
        f'--stats-file="{stats_file}"',
        f'--dump-config="{config_ini}"',
        f'"{SE_SCRIPT}"',
        '--cpu-type=X86O3CPU',
        '--num-cpus=1',
        '--caches',
        f'--l1i_size={int(row["l1i_kb"])}kB',
        f'--l1d_size={int(row["l1d_kb"])}kB',
        '--l2cache',
        f'--l2_size={int(row["l2_kb"])}kB',
        f'--cpu-clock={row["cpu_clock_GHz"]}GHz',
        f'--bp-type={row["branch_predictor"]}',
        f'-c "{WORKLOAD}"'
    ] + p_flags

    cmd = " \\\n  ".join(cmd_lines)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

    # Generate McPAT XML
    TEMPLATE = os.path.join(PARSER_DIR, "templates", f"template_x86_{row['branch_predictor']}.xml")
    subprocess.run([
        sys.executable,
        os.path.join(PARSER_DIR, "Gem5McPATParser.py"),
        "--config", config_json,
        "--stats", stats_file,
        "--template", TEMPLATE,
        "--output", mcpat_xml
    ], check=True)

    # Run McPAT
    with open(mcpat_out, "w") as f:
        subprocess.run([MCPAT_BIN, "-infile", mcpat_xml, "-print_level", "5"], stdout=f, check=True)

    return stats_file, mcpat_out

# ================= PER-CASE PROCESS =================
def process_case(row, csv_file, write_header=False):
    try:
        case_id = int(row["exp_id"])
        stats_file, mcpat_out = run_gem5_mcpat(case_id, row)
        mcpat_data = parse_mcpat_output(mcpat_out, row)
        stats_data = parse_stats_file(stats_file)
        combined = {**mcpat_data, **stats_data}
        for k, v in combined.items():
            if isinstance(v, (float, int)):
                combined[k] = round(v, 4)

        # Write to CSV immediately
        df_row = pd.DataFrame([combined])
        df_row.to_csv(csv_file, index=False, mode='a', header=write_header)
        print(f"Case {case_id} finished")
        return True
    except Exception as e:
        print(f"Error in case {row['exp_id']}: {e}")
        traceback.print_exc()
        return False


# ================= MAIN =================
def main():
    MAX_WORKERS = 30
    csv_file = "gem5_mcpat_stats_results.csv"

    try:
        df = pd.read_csv(csv_path)
        total_cases = len(df)
        print(f"Loaded {total_cases} cases from {csv_path}")

        if TEST_MODE:
            df = df.iloc[:1]
            total_cases = 1

        t_start = time.time()
        if os.path.exists(csv_file):
            os.remove(csv_file)

        first_write = True
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(df))) as executor:
            futures = {executor.submit(process_case, row, csv_file, first_write): idx for idx, row in df.iterrows()}
            for future in as_completed(futures):
                res = future.result()
                if first_write:
                    first_write = False 

        t_end = time.time()
        print("\n================ TIMING REPORT ================")
        print(f"Total time: {(t_end-t_start)/60:.2f} minutes for {total_cases} cases")
        print(f"Average time per case {(t_end-t_start)/60/total_cases:.2f}")
        print("===============================================")

    except Exception:
        print("\nAn error occurred during processing:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
