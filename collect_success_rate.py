import argparse
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Collect success rates from eval logs")
    parser.add_argument("path", type=str, help="e.g. eval_logs/libero_10/algo_tt_update0_num_step_tt_in_traj0_720steps_eps50_ah8")
    args = parser.parse_args()

    root = Path(args.path)
    results = {}
    missing = []

    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        txt_file = task_dir / f"{task_dir.name}.txt"
        if not txt_file.exists():
            txt_files = list(task_dir.glob("*.txt"))
            if not txt_files:
                missing.append(task_dir.name)
                continue
            txt_file = txt_files[0]

        rate = None
        for line in reversed(txt_file.read_text().splitlines()):
            m = re.search(r"success rate:\s*([\d.]+)", line)
            if m:
                rate = float(m.group(1))
                break

        if rate is None:
            missing.append(task_dir.name)
        else:
            results[task_dir.name] = rate

    for name, rate in results.items():
        print(f"{rate:.2f}  {name}")

    if missing:
        print(f"\nWARNING: no success rate found for {len(missing)} task(s):")
        for name in missing:
            print(f"  {name}")

    if results:
        mean = sum(results.values()) / len(results)
        print(f"\nTasks: {len(results)}")
        print(f"Mean success rate: {mean:.4f}")
    else:
        print("No success rates found.")

if __name__ == "__main__":
    main()
