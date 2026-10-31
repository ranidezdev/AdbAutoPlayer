"""AFK Journey Sunlit Showdown Mixin."""

import logging
from abc import ABC
from time import sleep

from adb_auto_player.decorators import register_command, register_custom_routine_choice
from adb_auto_player.games.afk_journey.gui_category import AFKJCategory
from adb_auto_player.games.afk_journey.mixins._showdown_event import (
    _ShowdownEventMixin,
)
from adb_auto_player.models import ConfidenceValue
from adb_auto_player.models.decorators import GUIMetadata
from adb_auto_player.models.geometry import Point


class SunlitShowdownMixin(_ShowdownEventMixin, ABC):
    """Sunlit Showdown Mixin."""

    _EVENT_DIR = "event/sunlit_showdown"
    _EVENT_NAME = "Sunlit Showdown"

    @register_command(
        name="SunlitShowdown",
        gui=GUIMetadata(
            label="Run Sunlit Showdown",
            category=AFKJCategory.EVENTS_AND_OTHER,
            tooltip="Participate in the Sunlit Showdown event automatically",
        ),
    )
    @register_custom_routine_choice(label="Sunlit Showdown")
    def attempt_sunlit(self) -> None:
        """Attempt to run Sunlit Showdown Battles."""
        self.start_up()
        self.navigate_to_world()

        logging.info("Attempting to run Sunlit Showdown!")
        logging.warning(
            "Very much an alpha function, designed only to clear the shop. "
            "Currently checks for Shakir, Harak, Shemira, Lenya, Sonja, "
            "Vala, Faramor, Sinbad, Berial and Valka, and picks them in that priorty "
            "order as it scrolls down through heroes"
        )

        if self._open_sunlit_showdown() is False:
            return

        self._run_showdown_battles()

        logging.info("Sunlit Showdown Finished!")

    def _open_sunlit_showdown(self) -> bool:
        return self._open_showdown_event()

    def _select_heroes(self) -> None:
        self._handle_hero_selection()

    def _handle_hero_selection(self):
        selected_heroes = 0
        hero_slots = 3
        scrolls = 1
        max_scrolls = 5

        heroes = [
            "heroes/shakir_battle",
            "heroes/harak_battle",
            "heroes/shemira_battle",
            "heroes/lenya_battle",
            "heroes/sonja_battle",
            "heroes/vala_battle",
            "heroes/faramor_battle",
            "heroes/natsu_battle",
            "heroes/sinbad_battle",
            "heroes/berial_battle",
            "heroes/valka_battle",
        ]

        while selected_heroes < hero_slots:
            hero_checker = self.find_any_template(
                templates=heroes,
                threshold=ConfidenceValue(
                    "93%"
                ),  # Else Faramor triggers while already selected
            )
            if hero_checker is not None:
                scrolls = 1
                logging.info(
                    "Selecting "
                    + hero_checker.template.split("/")[1].split("_")[0].capitalize()
                )
                self.tap(hero_checker)
                sleep(1)
                selected_heroes += 1
                # logging.info(selected_heroes)
            elif scrolls <= max_scrolls:
                logging.info(
                    "No listed hero found, scolling down " + str(scrolls) + "/5"
                )
                self.swipe_up(630, 1630, 1340, duration=2)
                scrolls += 1
            if scrolls > max_scrolls:
                self.tap(Point(1000, 1625))
                self.tap(Point(715, 1600))
                self.tap(Point(850, 1600))
                self.tap(Point(1000, 1450))
                logging.info("Hero selection reset")
                return

    def _handle_battle_result(self) -> bool:
        result = self.wait_for_any_template(
            templates=["event/sunlit_showdown/victory", "event/sunlit_showdown/defeat"],
            timeout=self.BATTLE_TIMEOUT,
            delay=1.0,
            timeout_message=self.BATTLE_TIMEOUT_ERROR_MESSAGE,
        )

        match result.template:
            case "event/sunlit_showdown/victory":
                logging.info("Victory!")
                self.tap(self._TAP_TO_CLOSE_POINT)  # Tap to Close
                sleep(5)
                # If we won the final clear Battle Record and use next token
                if self.game_find_template_match(
                    template="event/sunlit_showdown/battle_record"
                ):
                    self.tap(self._TAP_TO_CLOSE_POINT)  # Tap Battle Record
                    sleep(5)
                    return self._start_next_run()
                return True

            case "event/sunlit_showdown/defeat":
                logging.warning("Defeat!")
                self.tap(self._TAP_TO_CLOSE_POINT)  # Tap to Close
                sleep(5)
                self.tap(self._TAP_TO_CLOSE_POINT)  # Tap Battle Record
                sleep(5)
                # Use another token
                return self._start_next_run(settle_seconds=4)

        logging.warning("Something went wrong detecting battle results!")
        self.capture_debug_screenshot("sunlit_showdown_battle_result_unknown")
        return False
