import os
import sys


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GUI_Panel.app import main


if __name__ == "__main__":
    main()
