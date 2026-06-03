import shutil, subprocess, os
docs = r'e:\SGLab\cable-driven\synergy_hand_sim\docs'

# Copy to simple name
shutil.copy2(os.path.join(docs, 'chapter3_section2.xdv'), os.path.join(docs, 'simple.xdv'))

# Convert
result = subprocess.run(
    ['xdvipdfmx', '-o', os.path.join(docs, 'simple.pdf'), os.path.join(docs, 'simple.xdv')],
    capture_output=True, text=True
)
print('Return code:', result.returncode)
print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)

# Check
pdf_path = os.path.join(docs, 'simple.pdf')
print('PDF exists:', os.path.exists(pdf_path))
if os.path.exists(pdf_path):
    print('PDF size:', os.path.getsize(pdf_path))

# List all output files
for f in os.listdir(docs):
    if 'simple' in f:
        print('File:', f)
