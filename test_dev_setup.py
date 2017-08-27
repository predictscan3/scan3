from __future__ import print_function

import settings

from os.path import exists

print("Data folder exists? {0}".format(exists(settings.DATA_ROOT)))