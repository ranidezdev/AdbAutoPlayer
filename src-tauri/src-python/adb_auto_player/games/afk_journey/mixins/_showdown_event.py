"""Shared flow for the Showdown battle events (Sunlit / Frostfire)."""

import logging
from abc import ABC, abstractmethod
from time import sleep
from typing import ClassVar

from adb_auto_player.games.afk_journey.base import AFKJourneyBase
from adb_auto_player.models.geometry import Point


class _ShowdownEventMixin(AFKJourneyBase, ABC):
    """Common navigation and battle loop of the Showdown events.

    Subclasses provide the event template directory, the hero selection and
    the battle-result handling, which differ between the events.
    """

    # e.g. "event/sunlit_showdown"
    _EVENT_DIR: ClassVar[str]
    _EVENT_NAME: ClassVar[str]

    _JOIN_POINT = Point(650, 1450)
    _START_POINT = Point(800, 1800)
    _TAP_TO_CLOSE_POINT = Point(550, 1800)
    _HERO_SLOT_POINTS: ClassVar[tuple[Point, ...]] = (
        Point(425, 950),
        Point(325, 870),
        Point(175, 870),
    )

    @abstractmethod
    def _select_heroes(self) -> None:
        """Fill the three empty hero slots."""

    @abstractmethod
    def _handle_battle_result(self) -> bool:
        """Handle the victory/defeat screen.

        Returns:
            True if another battle should be fought.
        """

    def _on_run_screen(self) -> None:
        """Hook called on the event screen right before a run is started."""

    def _event_template(self, name: str) -> str:
        return f"{self._EVENT_DIR}/{name}"

    def _run_showdown_battles(self) -> None:
        while self._handle_battle():
            self._handle_battle()

    def _open_showdown_event(self) -> bool:
        """Open the event from the hamburger menu and start the first run.

        Returns:
            False if no event tokens are left.
        """
        logging.info("Opening %s", self._EVENT_NAME)
        event_entry = self._EVENT_DIR.rsplit("/", maxsplit=1)[-1]
        self._navigate_menu_chain(
            [
                "navigation/hamburger_menu",
                "dailies/hamburger/events",
                self._event_template(event_entry),
            ]
        )
        self.wait_for_template(template=self._event_template("title_s"))
        self.tap(self._JOIN_POINT)  # Join
        sleep(3)
        return self._start_next_run(settle_seconds=4)

    def _start_next_run(self, settle_seconds: float = 0) -> bool:
        """Start the next run (uses an event token) from the event screen.

        Args:
            settle_seconds: Seconds to wait after tapping Start before checking
                for the out-of-tokens popup.

        Returns:
            False if no event tokens are left.
        """
        self.wait_for_template(template=self._event_template("title_s"))
        self._on_run_screen()
        self.tap(self._START_POINT)  # Start/Continue
        if settle_seconds:
            sleep(settle_seconds)
        if self.game_find_template_match(
            template=self._event_template("insufficient_resources")
        ):
            logging.warning("Out of %s tokens!", self._EVENT_NAME)
            return False
        return True

    def _handle_battle(self) -> bool:
        # Enter hero selection screen
        self.wait_for_template(template=self._event_template("quick_select"))
        self.tap(self._START_POINT)  # Continue/Battle
        sleep(2)
        self._tap_till_template_disappears(template="navigation/confirm")  # consumables
        sleep(4)

        # Clear all hero spots
        self.wait_for_template(template="start_battle")
        logging.debug("Clearing heroes")
        for slot in self._HERO_SLOT_POINTS:
            self.tap(slot)
            sleep(1)

        self._select_heroes()

        # Start Battle
        self._tap_till_template_disappears(template="start_battle")  # Battle
        logging.info("Battling..")
        sleep(3)
        self._tap_till_template_disappears(template="navigation/confirm")  # Confirm
        sleep(15)

        return self._handle_battle_result()
