#!/usr/bin/env python3
"""Clean batch_analysis.py by removing broken dictionary code"""

with open('batch_analysis.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep lines 1-175 and lines 469 onwards
clean_lines = lines[:176] + ['\n'] + lines[469:]

with open('batch_analysis.py', 'w', encoding='utf-8') as f:
    f.writelines(clean_lines)

print("✅ Cleaned batch_analysis.py - removed lines 176-468 (broken dictionaries)")