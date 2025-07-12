#!/usr/bin/env python3
"""
HEDES Client - Command Line Interface

This is the entry point for the HEDES command-line client.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the application
from src.application import main

if __name__ == "__main__":
    # Set the working directory to the project root
    os.chdir(project_root)
    main()
