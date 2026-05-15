# src/automation/mouse_control.py
# ─────────────────────────────────────────────────────────────
# Low-level mouse control for the bot.
#
# WHY NOT JUST USE pyautogui DIRECTLY?
# We wrap it in a class so we can:
# 1. Add safety checks everywhere
# 2. Add logging to every action
# 3. Easily swap the implementation later
# 4. Add delays consistently
#
# SAFETY FIRST:
# pyautogui has a built-in failsafe — move mouse to any
# corner of the screen to emergency stop the bot.
# ─────────────────────────────────────────────────────────────

import pyautogui
import time
from src.utils.logger import get_logger

logger = get_logger("automation.mouse")

# ── SAFETY SETTINGS ───────────────────────────────────────────
# pyautogui failsafe: moving mouse to corner stops everything
pyautogui.FAILSAFE = True

# Minimum delay between pyautogui actions (seconds)
# Too fast = game doesn't register the action
pyautogui.PAUSE = 0.05


class MouseController:
    """
    Controls mouse movement and clicks for game automation.

    All coordinates are in SCREEN coordinates (absolute pixels).
    The caller must convert game coordinates to screen coordinates
    before calling these methods.
    """

    def __init__(
        self,
        move_duration: float = 0.3,
        drag_duration: float = 0.4
    ):
        """
        Args:
            move_duration: seconds to take when moving mouse
            drag_duration: seconds to take when dragging
        """
        self.move_duration = move_duration
        self.drag_duration = drag_duration
        logger.info(
            f"MouseController ready | "
            f"move={move_duration}s drag={drag_duration}s"
        )

    def move_to(self, x: int, y: int) -> None:
        """
        Smoothly moves mouse to (x, y) screen position.
        Uses pyautogui's built-in smooth movement.
        """
        logger.debug(f"Moving to ({x}, {y})")
        pyautogui.moveTo(
            x, y,
            duration=self.move_duration,
            tween=pyautogui.easeInOutQuad
        )

    def click(self, x: int, y: int) -> None:
        """
        Moves to position and clicks.
        """
        logger.debug(f"Clicking ({x}, {y})")
        pyautogui.click(x, y)

    def press(self, x: int, y: int) -> None:
        """
        Presses and HOLDS the mouse button at (x, y).
        Used to START a drag operation.
        """
        logger.debug(f"Pressing at ({x}, {y})")
        pyautogui.mouseDown(x, y)

    def release(self) -> None:
        """
        Releases the mouse button at current position.
        Used to END a drag operation (launch the bird).
        """
        pos = pyautogui.position()
        logger.debug(f"Releasing at {pos}")
        pyautogui.mouseUp()

    def drag_to(
        self,
        start_x: int,
        start_y: int,
        end_x:   int,
        end_y:   int
    ) -> None:
        """
        Performs a complete drag operation:
        1. Move to start position
        2. Press and hold
        3. Slowly drag to end position
        4. Release

        Args:
            start_x, start_y: where to press down (bird position)
            end_x, end_y:     where to drag to (pull-back position)
        """
        logger.info(
            f"Drag: ({start_x},{start_y}) → ({end_x},{end_y})"
        )

        # Step 1: move to start without clicking
        self.move_to(start_x, start_y)
        time.sleep(0.1)  # small pause before pressing

        # Step 2: press and hold
        pyautogui.mouseDown()
        time.sleep(0.15)  # hold briefly before dragging

        # Step 3: drag smoothly to pull-back position
        pyautogui.moveTo(
            end_x, end_y,
            duration=self.drag_duration,
            tween=pyautogui.easeInOutQuad
        )
        time.sleep(0.1)  # hold at pulled-back position briefly

        # Step 4: release to launch
        pyautogui.mouseUp()
        logger.info("Released — bird launched!")

    def get_position(self):
        """Returns current mouse position as (x, y)."""
        return pyautogui.position()

    def emergency_stop(self):
        """
        Moves mouse to top-left corner to trigger failsafe.
        This immediately stops all pyautogui operations.
        """
        logger.warning("EMERGENCY STOP triggered!")
        pyautogui.moveTo(0, 0)