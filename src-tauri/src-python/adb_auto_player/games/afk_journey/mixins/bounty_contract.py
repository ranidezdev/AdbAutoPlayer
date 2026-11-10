"""AFK Journey Bounty Contract Mixin."""

import logging
import re
from time import monotonic, sleep

from adb_auto_player.decorators import register_command, register_custom_routine_choice
from adb_auto_player.exceptions import AutoPlayerError, AutoPlayerWarningError
from adb_auto_player.models.decorators import GUIMetadata
from adb_auto_player.models.geometry import Point
from adb_auto_player.models.template_matching import TemplateMatchResult
from adb_auto_player.ocr import RapidOCRBackend
from adb_auto_player.util import SummaryGenerator

from ..base import AFKJourneyBase
from ..gui_category import AFKJCategory

# Matches "13/15" in "13/15 (x1 regenerates in 03:58:07)".
_COUNTER_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")


def parse_search_targets_counter(text: str) -> int | None:
    """Return the remaining Search Targets from the OCR'd counter line.

    Args:
        text: OCR text of the line below the Search Targets label.

    Returns:
        The remaining searches, or None if no "x/y" counter was found.
    """
    match = _COUNTER_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


class BountyContractMixin(AFKJourneyBase):
    """Bounty Contract Mixin.

    Bounty Contract is a Guild Mode reached from Battle Modes > Guild Mode.
    Its "Search Targets" button spends one of the daily searches (e.g. 13/15);
    this taps it until the counter reaches 0.
    """

    _bounty_contract_ocr_backend: RapidOCRBackend | None = None

    # x1, y1, x2, y2 of the counter line, e.g. "13/15 (x1 regenerates in ...)".
    _COUNTER_BOX = (400, 1815, 1070, 1880)
    # The "Tap to close" label of the result popups sits on top of the Search
    # Targets button; tapping this neutral spot instead can't start a search
    # if the popup closed in the meantime.
    _POPUP_CLOSE_POINT = Point(540, 150)
    # Fall back to tapping the "Tap to close" label itself every n-th attempt.
    _POPUP_TAPS_BEFORE_LABEL_TAP = 3
    # A search takes ~5s before the button can be used again.
    _SEARCH_TIMEOUT = 30.0
    _POLL_INTERVAL = 1.0
    _MAX_FAILED_SEARCHES = 3

    @register_command(
        name="BountyContract",
        gui=GUIMetadata(
            label="Bounty Contract",
            category=AFKJCategory.GAME_MODES,
            tooltip="Use all Bounty Contract Search Targets in Guild Mode",
        ),
    )
    @register_custom_routine_choice(label="Bounty Contract")
    def run_bounty_contract(self) -> None:
        """Use all Bounty Contract Search Targets."""
        self.start_up()
        try:
            self.navigate_to_bounty_contract_screen()
            self._use_all_search_targets()
        except AutoPlayerWarningError as e:
            logging.warning(f"{e}")
        except AutoPlayerError as e:
            logging.error(f"{e}")

    def _use_all_search_targets(self) -> None:
        state = self._wait_for_search_targets(timeout=self.template_timeout)
        if state is None:
            self.capture_debug_screenshot("bounty_contract_counter_not_found")
            raise AutoPlayerWarningError("Could not read the Search Targets counter.")

        remaining, button = state
        failed_searches = 0
        while remaining > 0:
            logging.info(f"Bounty Contract: Search Targets ({remaining} left)")
            self.tap(button)

            state = self._wait_for_search_targets(
                timeout=self._SEARCH_TIMEOUT, below=remaining
            )
            if state is None:
                failed_searches += 1
                if failed_searches >= self._MAX_FAILED_SEARCHES:
                    self.capture_debug_screenshot("bounty_contract_search_stuck")
                    raise AutoPlayerWarningError(
                        "Search Targets counter did not go down, stopping."
                    )
                logging.warning("Search Targets counter did not go down, retrying")
                state = self._wait_for_search_targets(timeout=self.template_timeout)
                if state is None:
                    self.capture_debug_screenshot("bounty_contract_screen_lost")
                    raise AutoPlayerWarningError("Bounty Contract screen not found.")
                remaining, button = state
                continue

            failed_searches = 0
            SummaryGenerator.increment(
                "Bounty Contract", "Searches", count=remaining - state[0]
            )
            remaining, button = state

        logging.info("Bounty Contract: all Search Targets used (0 left)")

    def _wait_for_search_targets(
        self, timeout: float, below: int | None = None
    ) -> tuple[int, TemplateMatchResult] | None:
        """Wait until Search Targets can be tapped and return its counter.

        Result popups ("Tap to close") covering the screen are dismissed. A
        grey (depleted) Search Targets button means no searches are left.

        Args:
            timeout: Seconds to wait.
            below: Only accept a counter lower than this, i.e. wait for a
                running search to finish.

        Returns:
            The remaining searches and the Search Targets button, or None if
            the button with a (lower) counter didn't show up in time.
        """
        # Popups first: the button can still match faintly behind them.
        templates = [
            *self.BOUNTY_CONTRACT_TAP_TO_CLOSE_TEMPLATES,
            self.BOUNTY_CONTRACT_SEARCH_TARGETS_TEMPLATE,
            self.BOUNTY_CONTRACT_DEPLETED_TEMPLATE,
        ]
        deadline = monotonic() + timeout
        popup_taps = 0
        while True:
            screenshot = self.get_screenshot()
            result = self.find_any_template(
                templates,
                crop_regions=self.BOUNTY_CONTRACT_BOTTOM_CROP,
                screenshot=screenshot,
            )
            if result is None:
                pass
            elif result.template == self.BOUNTY_CONTRACT_DEPLETED_TEMPLATE:
                return 0, result
            elif result.template == self.BOUNTY_CONTRACT_SEARCH_TARGETS_TEMPLATE:
                remaining = self._read_search_targets_counter(screenshot)
                if remaining is not None and (below is None or remaining < below):
                    return remaining, result
            else:
                logging.debug(f"Bounty Contract: closing popup ({result.template})")
                popup_taps += 1
                if popup_taps % self._POPUP_TAPS_BEFORE_LABEL_TAP == 0:
                    self.tap(result)
                else:
                    self.tap(self._POPUP_CLOSE_POINT)

            if monotonic() >= deadline:
                return None
            sleep(self._POLL_INTERVAL)

    def _read_search_targets_counter(self, screenshot) -> int | None:
        if self._bounty_contract_ocr_backend is None:
            self._bounty_contract_ocr_backend = RapidOCRBackend.pp_ocr_v5_rec()
        x1, y1, x2, y2 = self._COUNTER_BOX
        text = self._bounty_contract_ocr_backend.extract_text(screenshot[y1:y2, x1:x2])
        remaining = parse_search_targets_counter(text)
        if remaining is None:
            logging.debug(f"Bounty Contract: unreadable counter {text!r}")
        return remaining
