import subprocess
import sys

try:
    # Run run_rag.py using the venv python
    process = subprocess.run(
        ['.venv\\Scripts\\python.exe', 'run_rag.py', '--output', 'outputs/rag_results_v5.csv'],
        capture_output=True,
        text=True
    )
    print(process.stdout)
    print(process.stderr)
except Exception as e:
    print(f'Error: {e}')
