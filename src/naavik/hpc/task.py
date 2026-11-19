"""`python -m training_framework.hpc.task`: the remote task entry point, for
every target (not only Slurm).

The module name is part of the execution protocol: rendered job scripts call
it, and an environment built from an older framework commit (dependency
mode pins the framework) has only this name. So it keeps its historical
place; the implementation lives in `naavik.launcher.remote.task`.
"""

import sys

from naavik.launcher.remote.task import main

if __name__ == "__main__":
    sys.exit(main())
