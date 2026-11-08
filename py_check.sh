#!/bin/bash
PY=/home/chocheng9/images_reduced/.venv/bin/python
$PY - <<'PY'
import sklearn, pandas, numpy, torch
print(sklearn.__version__, pandas.__version__, numpy.__version__, torch.__version__)
PY
