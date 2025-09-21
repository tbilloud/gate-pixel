# Runs all examples at once, used for testing purposes

import subprocess
from pathlib import Path
import os

examples_dir = Path('examples')
for script in examples_dir.glob('*.py'):
    if script.name in ['run_all.py','compare_measure.py']:
        continue
    print('Running',script)
    env = dict(**os.environ)
    env["MPLBACKEND"] = "Agg"
    env["NAPARI_HEADLESS"] = "1"
    result = subprocess.run(
        ['python', script],
        capture_output=True,
        text=True,
        env=env
    )
    if result.returncode != 0:
        print(f"Error while running {script}")
        print(result.stderr)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    print('-'*200)