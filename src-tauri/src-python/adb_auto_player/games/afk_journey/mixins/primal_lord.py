"""AFK Journey Primal Lord Mixin."""

import logging
from abc import ABC

from adb_auto_player.decorators import register_command, register_custom_routine_choice
from adb_auto_player.exceptions import GameTimeoutError
from adb_auto_player.games.afk_journey.base import AFKJourneyBase
from adb_auto_player.games.afk_journey.gui_category import AFKJCategory
from adb_auto_player.models.decorators import GUIMetadata
from adb_auto_player.models.geometry import Point


class PrimalLordMixin(AFKJourneyBase, ABC):
    """Primal Lord Mixin."""

    # Events tab button at the bottom of the Battle Modes screen.
    _EVENTS_TAB: Point = Point(835, 1655)
    # Daily attempt cap for the Primal Lord.
    _MAX_ATTEMPTS: int = 3
    # Scroll attempts while searching for the Primal Lord tile in the Events grid.
    _MAX_SCROLLS: int = 5
    # The character auto-runs to the Primal Lord; allow enough time to arrive.
    _TRAVEL_TIMEOUT: float = 60.0

    @register_command(
        name="PrimalLord",
        gui=GUIMetadata(
            label="Primal Lord",
            category=AFKJCategory.EVENTS_AND_OTHER,
            tooltip="Challenge the Primal Lord until the daily attempts are used up",
        ),
    )
    @register_custom_routine_choice(label="Primal Lord")
    def run_primal_lord(self) -> None:
        """Run Primal Lord battles until the daily attempts are exhausted."""
        self.start_up(device_streaming=False)

        if not self._open_primal_lord():
            return

        if not self._challenge():
            return

        self._confirm_teleport()

        if not self._tap_swords_button():
            return

        self._run_battles()

        logging.info("Primal Lord finished.")

    ############################## Helper Functions ##############################

    def _open_primal_lord(self) -> bool:
        """Open the Primal Lord event page from the Events tab.

        Returns:
            True if the Primal Lord event was opened, False otherwise.
        """
        logging.info("Navigating to Primal Lord...")
        self.navigate_to_battle_modes_screen()
        self.tap(self._EVENTS_TAB, log_message="Opening Events tab")
        self.sleep_navigation()

        tile = self._find_primal_lord_tile()
        if tile is None:
            logging.warning(
                "Could not find Primal Lord. Is the event currently available?"
            )
            return False

        self.tap(tile, log_message="Selecting Primal Lord")
        self.sleep_navigation()
        return True

    def _find_primal_lord_tile(self):
        """Search the Events grid for the Primal Lord tile, scrolling if needed."""
        for _ in range(self._MAX_SCROLLS):
            if match := self.game_find_template_match("primal_lord/label.png"):
                return match
            self.swipe_up(sy=1350, ey=800)
            self.sleep_navigation()
        return self.game_find_template_match("primal_lord/label.png")

    def _challenge(self) -> bool:
        """Tap the Challenge button on the Primal Lord event page.

        Returns:
            True if the Challenge button was tapped, False otherwise.
        """
        if not self._try_wait_and_tap(
            "primal_lord/challenge.png",
            timeout=self.min_timeout,
            timeout_message="Challenge button not found.",
        ):
            logging.warning("Challenge button not found.")
            return False
        self.sleep_navigation()
        return True

    def _confirm_teleport(self) -> None:
        """Confirm the teleport popup if it appears.

        The popup only shows when the character is not already at the Primal Lord.
        """
        if self._try_wait_and_tap(
            "primal_lord/teleport_confirm.png",
            timeout=self.fast_timeout,
            timeout_message="No teleport popup, assuming already at the Primal Lord.",
        ):
            self.sleep_navigation()

    def _tap_swords_button(self) -> bool:
        """Wait for the character to reach the Primal Lord and tap the fight button.

        Returns:
            True if the swords button was tapped, False otherwise.
        """
        if not self._try_wait_and_tap(
            "primal_lord/swords.png",
            timeout=self._TRAVEL_TIMEOUT,
            timeout_message="Fight button (swords) not found.",
        ):
            logging.warning("Could not reach the Primal Lord.")
            return False
        self.sleep_navigation()
        return True

    def _run_battles(self) -> None:
        """Fight the Primal Lord until no attempts remain."""
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            if not self._try_wait_and_tap(
                "primal_lord/boss_battle.png",
                timeout=self.min_timeout,
                timeout_message="Battle button not found.",
            ):
                break
            self.sleep_navigation()

            try:
                self.wait_for_template(
                    "primal_lord/elite_challenge.png",
                    timeout=self.fast_timeout,
                    timeout_message="Formation screen not found.",
                )
            except GameTimeoutError:
                logging.info("No more Primal Lord attempts available.")
                self.press_back_button()
                break

            if not self._try_wait_and_tap(
                "primal_lord/battle.png",
                timeout=self.min_timeout,
                timeout_message="Battle button not found.",
            ):
                break

            logging.info(f"Primal Lord battle #{attempt} started.")
            self._wait_for_battle_end()

        logging.info("No more Primal Lord attempts.")

    def _wait_for_battle_end(self) -> None:
        """Wait for the battle to finish, then close the result screen."""
        try:
            self.wait_for_any_template(
                templates=["primal_lord/tap_to_close.png", "tap_to_close.png"],
                timeout=self.BATTLE_TIMEOUT,
                delay=1.0,
                timeout_message=self.BATTLE_TIMEOUT_ERROR_MESSAGE,
            )
        except GameTimeoutError:
            logging.warning("Battle end screen not found.")
            return

        self.tap(Point(540, 1770), log_message="Tap to close")
        self.sleep_navigation()
