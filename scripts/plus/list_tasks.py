"""Print the LIBERO-plus env ids of every task in a given perturbation category.

Usage: list_tasks.py "Background Textures" [suite ...]
With no suite given, every suite in task_classification.json is used.
"""

import json
import os
import sys

TASK_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../external_dependencies/LIBERO-plus/libero/libero/benchmark/task_classification.json",
)


def main():
    category = sys.argv[1]
    suites = sys.argv[2:]

    with open(TASK_JSON) as f:
        data = json.load(f)

    unknown = set(suites) - set(data)
    if unknown:
        sys.exit(f"unknown suite(s): {', '.join(sorted(unknown))}")

    categories = {t["category"] for tasks in data.values() for t in tasks}
    if category not in categories:
        sys.exit(f"unknown category '{category}'; available: {', '.join(sorted(categories))}")

    seen = set()
    for suite in suites or data:
        for task in data[suite]:
            name = task["name"]
            if task["category"] == category and name not in seen:
                seen.add(name)
                print(f"libero_sim/{name}")


if __name__ == "__main__":
    main()
