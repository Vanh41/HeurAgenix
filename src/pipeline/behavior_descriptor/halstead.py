import sys
import os
import inspect
from radon.metrics import h_visit



heuristic_file = sys.argv[1]
with open(heuristic_file, 'r') as f:
        code = f.read()
        
#Analyze with radon.metrics.h_visit (Halstead)
metrics = h_visit(code)

total = metrics.total

print(f"{total.vocabulary}")

