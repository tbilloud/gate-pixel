# Runs all examples at once for testing purposes

# Set to False to see messages from global_log()
avoid_global_log = True

import subprocess
from pathlib import Path
import os

examples = list(Path('examples').glob('*.py'))
offline = [e for e in examples if e.name.startswith('offline_')]
others = [e for e in examples if not e.name.startswith('offline_')]
ordered = others + offline
for script in ordered:

    # Avoid some scripts:
    # - the current one would loop forever
    # - the comparison with measured data needs additional data
    if script.name in ['run_all.py','compare_measure.py']:
        continue
    print('Running',script)
    env = dict(**os.environ)
    env["MPLBACKEND"] = "Agg"
    env["NAPARI_HEADLESS"] = "1"
    result = subprocess.run(
        ['python', script],
        capture_output=avoid_global_log,
        text=True,
        env=env
    )
    if result.returncode != 0:
        print(f"Error while running {script}")
        print(result.stderr)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    print('-'*200)