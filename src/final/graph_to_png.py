from subprocess import check_call
from bld.project_paths import project_paths_join as ppj

check_call(['dot', '-Tpng', ppj("OUT_FIGURES", "dag"), '-o', ppj("PROJECT_ROOT", "dag.png")])
