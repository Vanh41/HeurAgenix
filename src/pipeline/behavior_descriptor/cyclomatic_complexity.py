import inspect
import sys
from radon.complexity import cc_visit


heuristic_file = sys.argv[1]
with open(heuristic_file, 'r') as f:
        code = f.read()

# Analyze Cyclomatic Complexity
complexities = cc_visit(code=code)
obj = complexities[0]
print(f"{obj.complexity}")