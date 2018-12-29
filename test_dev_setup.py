from __future__ import print_function

from scan3 import settings

from os.path import exists
from sys import exit

print("Project root is {}".format(settings.PROJECT_ROOT))
print("Data IN folder is {}".format(settings.DATA_IN_ROOT))
print("Data OUT folder is {}".format(settings.DATA_OUT_ROOT))

ok = exists(settings.DATA_IN_ROOT)
print("Data folder exists? {0}".format(ok))

if not ok:
    exit(-1)
