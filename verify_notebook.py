#!/usr/bin/env python3
"""Verify the notebook by executing cells in the Oracle execution order.

Simulates the restart protocol:
1. Main Problem: Solution cell -> Testing Template cell
2. (restart)
3. Subproblem 1: Solution cell -> Testing Template cell
4. (restart)
5. Subproblem 2: Solution cell -> Testing Template cell
"""
import nbformat
import json
import subprocess
import sys
import tempfile
import os

nb_path = '/workspace/SC_Math-001.ipynb'
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

code_cells = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        code_cells.append((i, cell.source))

# Find cell indices by scanning markdown headers
def find_section_cells(nb):
    """Return dict mapping section names to lists of code cell indices."""
    current_section = None
    sections = {}
    code_idx = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            src = cell.source.lower()
            if '## subproblem 1' in src and 'subproblem' in src:
                current_section = 'sp1'
            elif '## subproblem 2' in src and 'subproblem' in src:
                current_section = 'sp2'
            elif '## main problem' in src:
                current_section = 'main'
            elif '### solution' in src:
                if current_section:
                    sections.setdefault(current_section, {})['solution_next'] = True
            elif '### testing template' in src:
                if current_section:
                    sections.setdefault(current_section, {})['testing_next'] = True
            elif '### gemini' in src:
                if current_section:
                    sections.setdefault(current_section, {})['gemini_next'] = True
        elif cell.cell_type == 'code':
            if current_section:
                sec = sections.setdefault(current_section, {})
                if sec.get('solution_next'):
                    sec['solution_cell'] = i
                    sec['solution_next'] = False
                elif sec.get('testing_next'):
                    sec['testing_cell'] = i
                    sec['testing_next'] = False
                elif sec.get('gemini_next'):
                    sec.setdefault('gemini_cells', []).append(i)
                    sec['gemini_next'] = False
    return sections

sections = find_section_cells(nb)

def run_cells(cell_indices, label):
    """Execute cells in order using a fresh Python process."""
    code_parts = []
    for idx in cell_indices:
        cell = nb.cells[idx]
        if cell.cell_type == 'code':
            src = cell.source
            if src.startswith('!') or src.startswith('import subprocess'):
                continue
            code_parts.append(f"# === Cell {idx} ===")
            code_parts.append(src)

    script = '\n\n'.join(code_parts)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=120
        )
        print(f"\n{'='*60}")
        print(f"=== {label} ===")
        print(f"{'='*60}")
        if result.stdout:
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr[-1000:]}")
        print(f"Exit code: {result.returncode}")
        return result.returncode == 0
    finally:
        os.unlink(tmp)

# Execute Oracle protocol
all_pass = True

# Main Problem: Solution -> Testing
if 'main' in sections:
    s = sections['main']
    ok = run_cells([s['solution_cell'], s['testing_cell']], "Main Problem Oracle")
    all_pass = all_pass and ok

# Subproblem 1: Solution -> Testing
if 'sp1' in sections:
    s = sections['sp1']
    ok = run_cells([s['solution_cell'], s['testing_cell']], "Subproblem 1 Oracle")
    all_pass = all_pass and ok

# Subproblem 2: Solution -> Testing
if 'sp2' in sections:
    s = sections['sp2']
    ok = run_cells([s['solution_cell'], s['testing_cell']], "Subproblem 2 Oracle")
    all_pass = all_pass and ok

print(f"\n{'='*60}")
print(f"OVERALL ORACLE RESULT: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print(f"{'='*60}")
