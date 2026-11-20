"""`python -m naavik.launcher.task`: the task entry point a plan of
execution protocol 2 runs on the host, for every scheduler. Plans of
protocol 1 run `training_framework.hpc.task`, which still exists
(`naavik.hpc.task`, through the `training_framework` alias). The
implementation lives in `naavik.launcher.remote.task`.
"""

import sys

from naavik.launcher.remote.task import main

if __name__ == "__main__":
    sys.exit(main())
