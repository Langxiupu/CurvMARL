import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.run_mappo import main

if __name__ == '__main__':
    main()
