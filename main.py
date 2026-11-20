"""Run the game:   python main.py          (optional: python main.py <map seed>)"""
import multiprocessing

from diamondquest.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
