from pathlib import Path  


base_path = Path('results')
for path in base_path.iterdir():
    if len(list(path.iterdir())) == 0:
        path.rmdir()