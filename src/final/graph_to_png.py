from graphviz import render
from graphviz import Source
from bld.project_paths import project_paths_join as ppj


# render('dot', 'png', '/home/christopher/Dokumente/rm_fused_lasso/bld/out/figures/dag')
# 'test-output/holy-grenade.gv.png'
#
# Source.from_file('/home/christopher/Dokumente/rm_fused_lasso/bld/out/figures/dag')

from subprocess import check_call
check_call(['dot','-Tpng',ppj("OUT_FIGURES", "dag"),'-o',ppj("PROJECT_ROOT", "project_graph.png")])
