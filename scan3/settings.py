from scan3.conf import *

# Developer specific settings and configuration here
# Most things should be common though, so should be in conf.py

from os.path import join, exists
from os import makedirs

DATA_OUT_ROOT = join(PROJECT_ROOT, "tmp")

if not exists(DATA_OUT_ROOT):
    print("Creating output directory: {}".format(DATA_OUT_ROOT))
    makedirs(DATA_OUT_ROOT)
