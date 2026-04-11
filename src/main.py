"""
main.py
Entrypoint for the Blink Mouse Control demo.
Run this file to start the program.
"""

from blink_mouse_control import run_detection

def main():
    print("Blink Mouse Control - starting...")
    run_detection()
    print("Program ended.")

if __name__ == "__main__":
    main()


