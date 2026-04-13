"""Mouse action utilities for blink events."""

import time

import pyautogui


class MouseActions:
    """Encapsulate mouse operations triggered by blink events."""

    def left_click(self) -> None:
        pyautogui.click(button="left")

    def right_click(self) -> None:
        pyautogui.click(button="right")

    def hold_left_click(self, hold_seconds: float) -> None:
        pyautogui.mouseDown()
        time.sleep(hold_seconds)
        pyautogui.mouseUp()

    def scroll_up(self, amount: int) -> None:
        pyautogui.scroll(max(1, int(amount)))

    def scroll_down(self, amount: int) -> None:
        pyautogui.scroll(-max(1, int(amount)))


class NoOpMouseActions(MouseActions):
    """Mouse action adapter that safely ignores all click requests."""

    def left_click(self) -> None:
        return

    def right_click(self) -> None:
        return

    def hold_left_click(self, hold_seconds: float) -> None:
        return

    def scroll_up(self, amount: int) -> None:
        return

    def scroll_down(self, amount: int) -> None:
        return
