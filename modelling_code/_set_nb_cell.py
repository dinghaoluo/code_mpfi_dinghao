import json
import sys
from pathlib import Path

nb_path = Path(sys.argv[1])
cell_index = int(sys.argv[2])
text_path = Path(sys.argv[3])

nb = json.loads(nb_path.read_text(encoding='utf-8'))
text = text_path.read_text(encoding='utf-8')
nb['cells'][cell_index]['source'] = text.splitlines(True)
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'updated cell {cell_index}')
