"""Legacy script entrypoint for local execution.

This file is intentionally small and delegates execution to the package CLI.
"""

from blink_mouse_control.cli import main


if __name__ == "__main__":
    main()


