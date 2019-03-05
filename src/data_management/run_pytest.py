from subprocess import call
from bld.project_paths import project_paths_join as ppj

call(['pytest', '-v'], cwd=ppj("IN_DATA_MANAGEMENT"))





#Popen(['pytest'], shell=True)

#print(call(['pytest','-v']))

#print(call(['cd','/home/christopher/Dokumente/rm_fused_lasso/src/model_code', '|', 'ls', '-l']))
