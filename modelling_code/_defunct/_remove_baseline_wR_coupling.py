import json
from pathlib import Path

nb_path = Path("modelling_code/general_model_v2.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cell2 = "".join(nb["cells"][2]["source"])
cell2 = cell2.replace(
    "    baseline_wR_coupling: float = 0.04  # lower-baseline cells receive slightly stronger run coupling, globally across the population\n",
    "",
)
nb["cells"][2]["source"] = cell2.splitlines(keepends=True)

cell4 = "".join(nb["cells"][4]["source"])
old = (
    "    wR_center = p.wR_mean + p.baseline_wR_coupling * baseline_z\n"
    "    wR = rng.normal(wR_center, p.wR_sd, p.n_cells)\n"
)
new = "    wR = rng.normal(p.wR_mean, p.wR_sd, p.n_cells)\n"
if old not in cell4:
    raise SystemExit('expected wR coupling block not found')
cell4 = cell4.replace(old, new)
nb["cells"][4]["source"] = cell4.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
