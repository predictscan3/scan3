from __future__ import print_function

from os.path import exists, dirname, basename, abspath, join

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
DATA_ROOT = join(PROJECT_ROOT, "data")

print("Project root is: {0}".format(PROJECT_ROOT))
print("Data root is: {0}".format(DATA_ROOT))

