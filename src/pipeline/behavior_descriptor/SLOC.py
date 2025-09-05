import os
import sys
import inspect
from radon.raw import analyze


heuristic_file = sys.argv[1]
with open(heuristic_file, 'r') as f:
        code = f.read()
        
# Analyze SLOC with radon.raw.analyze
raw_metrics = analyze(code)

# Print metrics
print(f"{raw_metrics.sloc}")
