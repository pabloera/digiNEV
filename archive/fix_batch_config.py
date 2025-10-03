#!/usr/bin/env python3
"""
Script to fix batch_analysis.py by removing hardcoded dictionaries
"""

import re

# Read the file
with open('batch_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the class definition
lines = content.split('\n')
new_lines = []
skip_mode = False
skip_start = 0

for i, line in enumerate(lines):
    # Keep lines before line 176
    if i < 175:
        new_lines.append(line)
    # Start skipping from line 176 (the broken dictionary content)
    elif i == 175:
        new_lines.append(line)  # Keep the return statement
        skip_mode = True
        skip_start = i + 1
    # Stop skipping when we find the MAIN ANALYZER CLASS comment
    elif skip_mode and '# MAIN ANALYZER CLASS' in line:
        skip_mode = False
        new_lines.append('')  # Add blank line
        new_lines.append(line)
    # Continue with normal lines after skipping
    elif not skip_mode:
        new_lines.append(line)

# Write the fixed file
with open('batch_analysis_fixed.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print(f"Fixed file created: batch_analysis_fixed.py")
print(f"Removed lines {skip_start} to {i-1}")