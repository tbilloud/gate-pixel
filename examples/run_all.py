# Runs all examples at once, used for testing purposes

# Set to True to see logs
avoid_log = True

import subprocess
from pathlib import Path
import os

examples = list(Path('examples').glob('*.py'))
offline = [e for e in examples if e.name.startswith('offline_')]
others = [e for e in examples if not e.name.startswith('offline_')]
ordered = others + offline
for script in ordered:
    if script.name in ['run_all.py','compare_measure.py']:
        continue
    print('Running',script)
    env = dict(**os.environ)
    env["MPLBACKEND"] = "Agg"
    env["NAPARI_HEADLESS"] = "1"
    result = subprocess.run(
        ['python', script],
        capture_output=avoid_log,
        text=True,
        env=env
    )
    if result.returncode != 0:
        print(f"Error while running {script}")
        print(result.stderr)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    print('-'*200)