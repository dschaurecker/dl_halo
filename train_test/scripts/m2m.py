import os
import sys

p = os.path.abspath('/scratch/ds6311/github/dl_halo/train_test/')
if p not in sys.path:
    sys.path.append(p)

from map2map.main import main


if __name__ == '__main__':
    main()
