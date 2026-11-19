"""`python -m naavik.hpc`: the launcher's command line, under
the name earlier versions documented. Imported (not run), this module is the
launcher's CLI module itself, so code that patches it patches the CLI."""

import sys

from naavik.launcher import __main__ as _cli

if __name__ == "__main__":
    sys.exit(_cli.main())
else:
    sys.modules[__name__] = _cli
