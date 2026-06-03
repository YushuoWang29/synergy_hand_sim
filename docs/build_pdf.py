import subprocess, os, sys

docs_path = r'e:\SGLab\cable-driven\synergy_hand_sim\docs'
tex_path = os.path.join(docs_path, 'chapter3_section2.tex')

# Step 1: Clean up old files
for f in ['chapter3_section2.pdf', 'chapter3_section2.xdv']:
    fp = os.path.join(docs_path, f)
    if os.path.exists(fp):
        os.remove(fp)
        print(f'Removed {f}')

# Step 2: Run xelatex (no-pdf mode) with encoding handling
print('\n=== Running xelatex (no-pdf) ===')
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

result = subprocess.run(
    ['xelatex', '-no-pdf', '-interaction=nonstopmode', tex_path],
    capture_output=True, text=True, cwd=docs_path,
    encoding='utf-8', errors='replace'
)
# Print last lines of stdout
lines = result.stdout.split('\n')
for line in lines[-30:]:
    print(line)
print('Xelatex return code:', result.returncode)

# Check for XDV
xdv_path = os.path.join(docs_path, 'chapter3_section2.xdv')
if os.path.exists(xdv_path):
    print(f'\nXDV created: {os.path.getsize(xdv_path)} bytes')
else:
    print('\nXDV NOT FOUND!')
    # list output files
    for f in os.listdir(docs_path):
        if 'chapter3' in f or 'ch3' in f:
            print('Found:', f)
    sys.exit(1)

# Step 3: Convert XDV to PDF
print('\n=== Converting XDV to PDF ===')
result = subprocess.run(
    ['xdvipdfmx', '-o', os.path.join(docs_path, 'chapter3_section2.pdf'), xdv_path],
    capture_output=True, text=True, cwd=docs_path,
    encoding='utf-8', errors='replace'
)
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr)
print('Return code:', result.returncode)

pdf_path = os.path.join(docs_path, 'chapter3_section2.pdf')
if os.path.exists(pdf_path):
    print(f'\nSUCCESS! PDF created: {os.path.getsize(pdf_path)} bytes')
else:
    print('\nFAILED: PDF not created')
