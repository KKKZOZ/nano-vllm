import sys

from calibrate import main

if __name__ == "__main__":
    print(
        "calibrate_aime.py is deprecated; use calibrate.py instead.",
        file=sys.stderr,
    )
    main()
