import json
from pathlib import Path

nb_path = Path('modelling_code/general_model_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

cell2 = ''.join(nb['cells'][2]['source']).replace('    run_off_mid: float    = 2.00  # midpoint of the falling sigmoid\n', '    run_off_mid: float    = 1.90  # midpoint of the falling sigmoid\n')
cell7 = ''.join(nb['cells'][7]['source']).replace("    (d, 'D', 'darkgreen'),\n", "    (d, 'D_release', 'darkgreen'),\n")

nb['cells'][2]['source'] = cell2.splitlines(keepends=True)
nb['cells'][7]['source'] = cell7.splitlines(keepends=True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
