#!/usr/bin/env python3
"""Verify that Gemini-3-Pro solutions FAIL the tests (as expected)."""
import nbformat
import subprocess
import sys
import tempfile
import os

nb_path = '/workspace/SC_Math-001.ipynb'
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)


def find_section_cells(nb):
    current_section = None
    sections = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            src = cell.source.lower()
            if '## subproblem 1' in src:
                current_section = 'sp1'
            elif '## subproblem 2' in src:
                current_section = 'sp2'
            elif '## main problem' in src:
                current_section = 'main'
            elif '### solution' in src and current_section:
                sections.setdefault(current_section, {})['solution_next'] = True
            elif '### testing template' in src and current_section:
                sections.setdefault(current_section, {})['testing_next'] = True
            elif '### gemini-3-pro (golden' in src and current_section:
                sections.setdefault(current_section, {})['golden_next'] = True
            elif '### gemini-3-pro' in src and current_section:
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
                elif sec.get('golden_next'):
                    sec['golden_cell'] = i
                    sec['golden_next'] = False
                elif sec.get('gemini_next'):
                    sec['gemini_cell'] = i
                    sec['gemini_next'] = False
    return sections

sections = find_section_cells(nb)

def run_cells(cell_indices, label):
    code_parts = []
    for idx in cell_indices:
        cell = nb.cells[idx]
        if cell.cell_type == 'code':
            src = cell.source
            if src.startswith('import subprocess'):
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
        out = result.stdout
        if len(out) > 3000:
            out = out[-3000:]
        print(out)
        if result.returncode != 0:
            print(f"STDERR (last 500 chars):\n{result.stderr[-500:]}")
        return result.returncode
    finally:
        os.unlink(tmp)


# SP1 Gemini: Run SP1 Gemini cell (which redefines function + runs tests)
# Need testing template defined first, then gemini overwrites function
if 'sp1' in sections:
    s = sections['sp1']
    # Need: testing helpers, gemini function, run tests
    # Gemini cell already includes test running code
    # But it needs the test functions defined first
    run_cells([s['testing_cell'], s['gemini_cell']],
              "Subproblem 1 Gemini (expect FAIL)")

# SP2 Gemini
if 'sp2' in sections:
    s = sections['sp2']
    run_cells([s['testing_cell'], s['gemini_cell']],
              "Subproblem 2 Gemini (expect FAIL)")

# Main Gemini: Need solution (for subproblem functions) + testing + gemini override
if 'main' in sections:
    s = sections['main']
    # The gemini cell needs: helpers, test definitions, subproblem functions, then gemini redefines main functions
    # The testing cell defines helpers and tests
    # The solution cell defines all functions
    # The gemini cell redefines compound/maximize functions
    # So: solution (defines SP funcs) -> testing (defines tests) -> gemini (overwrites + runs)
    run_cells([s['solution_cell'], s['testing_cell'], s['gemini_cell']],
              "Main Problem Gemini (expect FAIL)")
