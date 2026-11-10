"""Tests for the Bounty Contract mixin."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import adb_auto_player.games.afk_journey.mixins.bounty_contract as bounty_module
import cv2
import numpy as np
import pytest
from adb_auto_player.exceptions import AutoPlayerWarningError
from adb_auto_player.games.afk_journey.mixins.bounty_contract import (
    BountyContractMixin,
    parse_search_targets_counter,
)
from adb_auto_player.models import ConfidenceValue
from adb_auto_player.models.image_manipulation import CropRegions
from adb_auto_player.template_matching.template_matcher import TemplateMatcher

_SEARCH = BountyContractMixin.BOUNTY_CONTRACT_SEARCH_TARGETS_TEMPLATE
_DEPLETED = BountyContractMixin.BOUNTY_CONTRACT_DEPLETED_TEMPLATE


class _Stub(BountyContractMixin):
    """Minimal stub - only control-flow methods exercised."""

    def __init__(self) -> None:
        pass

    @property
    def template_timeout(self) -> float:
        return 5.0


class TestParseSearchTargetsCounter:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("15/15", 15),
            ("0/15", 0),
            ("13/15 ( x1 regenerates in 03:58:07)", 13),
            ("13/15\uff08\u56db xl regenerates in 03:58:07)", 13),
            ("7 / 15", 7),
            ("Search Targets", None),
            ("", None),
        ],
    )
    def test_parse(self, text, expected):
        assert parse_search_targets_counter(text) == expected


class TestRunBountyContract:
    def test_navigates_then_searches(self):
        bot = _Stub()
        manager = MagicMock()
        with (
            patch.object(bot, "start_up"),
            patch.object(bot, "navigate_to_bounty_contract_screen") as nav,
            patch.object(bot, "_use_all_search_targets") as use,
        ):
            manager.attach_mock(nav, "nav")
            manager.attach_mock(use, "use")
            bot.run_bounty_contract()

        assert manager.mock_calls == [call.nav(), call.use()]

    def test_warning_is_logged_not_raised(self):
        bot = _Stub()
        with (
            patch.object(bot, "start_up"),
            patch.object(bot, "navigate_to_bounty_contract_screen"),
            patch.object(
                bot,
                "_use_all_search_targets",
                side_effect=AutoPlayerWarningError("stuck"),
            ),
        ):
            bot.run_bounty_contract()


class TestUseAllSearchTargets:
    def _run(self, bot, wait_side_effect):
        mocks = {}
        with (
            patch.object(
                bot, "_wait_for_search_targets", side_effect=wait_side_effect
            ) as mocks["wait"],
            patch.object(bot, "tap") as mocks["tap"],
            patch.object(bot, "capture_debug_screenshot") as mocks["debug"],
            patch.object(bounty_module.SummaryGenerator, "increment") as mocks["sum"],
        ):
            bot._use_all_search_targets()
        return mocks

    def test_taps_until_counter_reaches_zero(self):
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        mocks = self._run(bot, [(2, button), (1, button), (0, button)])

        assert mocks["tap"].call_args_list == [call(button), call(button)]
        assert mocks["wait"].call_args_list[1].kwargs["below"] == 2
        assert mocks["wait"].call_args_list[2].kwargs["below"] == 1
        assert mocks["sum"].call_count == 2

    def test_no_tap_when_already_zero(self):
        bot = _Stub()
        mocks = self._run(bot, [(0, MagicMock())])

        mocks["tap"].assert_not_called()

    def test_counter_not_found_raises_warning(self):
        bot = _Stub()
        with pytest.raises(AutoPlayerWarningError):
            self._run(bot, [None])

    def test_retries_when_counter_did_not_go_down(self):
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        # tap -> no change -> screen re-checked -> tap again -> done
        mocks = self._run(bot, [(1, button), None, (1, button), (0, button)])

        assert mocks["tap"].call_count == 2

    def test_stops_after_repeated_failed_searches(self):
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        side_effect: list[tuple[int, MagicMock] | None] = [(3, button)]
        for _ in range(BountyContractMixin._MAX_FAILED_SEARCHES):
            side_effect += [None, (3, button)]

        with pytest.raises(AutoPlayerWarningError):
            self._run(bot, side_effect)

    def test_counts_multiple_searches_used_at_once(self):
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        mocks = self._run(bot, [(3, button), (1, button), (0, button)])

        assert [c.kwargs["count"] for c in mocks["sum"].call_args_list] == [2, 1]


class TestWaitForSearchTargets:
    def _run(self, bot, results, counters, *, timeout=10.0, below=None):
        """Run with one find_any_template result per poll."""
        mocks = {}
        clock = iter(range(1000))
        with (
            patch.object(bot, "get_screenshot", return_value=MagicMock()),
            patch.object(bot, "find_any_template", side_effect=results) as mocks[
                "find"
            ],
            patch.object(
                bot, "_read_search_targets_counter", side_effect=counters
            ) as mocks["ocr"],
            patch.object(bot, "tap") as mocks["tap"],
            patch.object(bounty_module, "sleep"),
            patch.object(bounty_module, "monotonic", side_effect=lambda: next(clock)),
        ):
            mocks["result"] = bot._wait_for_search_targets(timeout, below=below)
        return mocks

    def test_popups_are_checked_before_buttons(self):
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        mocks = self._run(bot, [button], [3])

        templates = mocks["find"].call_args.args[0]
        popup_count = len(bot.BOUNTY_CONTRACT_TAP_TO_CLOSE_TEMPLATES)
        assert templates[:popup_count] == list(
            bot.BOUNTY_CONTRACT_TAP_TO_CLOSE_TEMPLATES
        )
        assert templates[popup_count:] == [_SEARCH, _DEPLETED]

    def test_closes_popup_on_neutral_point_before_reading_counter(self):
        bot = _Stub()
        popup = MagicMock(template="tap_to_close.png")
        button = MagicMock(template=_SEARCH)
        mocks = self._run(bot, [popup, button], [12])

        assert mocks["tap"].call_args_list == [call(bot._POPUP_CLOSE_POINT)]
        assert mocks["result"] == (12, button)

    def test_taps_popup_label_when_neutral_taps_do_not_close_it(self):
        bot = _Stub()
        popup = MagicMock(template="tap_to_close.png")
        button = MagicMock(template=_SEARCH)
        taps = bot._POPUP_TAPS_BEFORE_LABEL_TAP
        mocks = self._run(bot, [popup] * taps + [button], [5])

        assert mocks["tap"].call_args_list[-1] == call(popup)
        assert mocks["tap"].call_count == taps

    def test_waits_for_counter_below_previous_value(self):
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        mocks = self._run(bot, [button] * 3, [5, 5, 4], below=5)

        assert mocks["result"] == (4, button)
        assert mocks["ocr"].call_count == 3

    def test_unreadable_counter_keeps_waiting(self):
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        mocks = self._run(bot, [button] * 2, [None, 0])

        assert mocks["result"] == (0, button)

    def test_grey_button_means_no_searches_left(self):
        # Regression: at 0/15 the button turns grey and the green template no
        # longer matches, which used to be treated as a stuck search.
        bot = _Stub()
        button = MagicMock(template=_SEARCH)
        depleted = MagicMock(template=_DEPLETED)
        mocks = self._run(bot, [button, depleted], [1], below=1)

        assert mocks["result"] == (0, depleted)
        assert mocks["ocr"].call_count == 1

    def test_returns_none_on_timeout(self):
        bot = _Stub()
        mocks = self._run(bot, [None] * 10, [], timeout=3)

        assert mocks["result"] is None
        mocks["tap"].assert_not_called()


class TestNavigateToBountyContractScreen:
    def test_returns_early_when_already_on_screen(self):
        bot = _Stub()
        with (
            patch.object(bot, "find_any_template", return_value=MagicMock()),
            patch.object(bot, "navigate_to_battle_modes_screen") as nav,
        ):
            bot.navigate_to_bounty_contract_screen()

        nav.assert_not_called()

    def test_opens_card_then_hall_board(self):
        bot = _Stub()
        card = MagicMock(template="battle_modes/bounty_contract.png")
        board = MagicMock(template="bounty_contract/hall_label.png")
        with (
            patch.object(bot, "find_any_template", return_value=None),
            patch.object(bot, "navigate_to_battle_modes_screen"),
            patch.object(bot, "_open_guild_mode_tab") as open_tab,
            patch.object(bot, "_find_in_battle_modes", return_value=card),
            patch.object(bot, "wait_for_template", return_value=board),
            patch.object(bot, "_tap_till_template_disappears") as tap_till,
            patch.object(bot, "wait_for_any_template") as wait_any,
            patch.object(bot, "sleep_navigation"),
        ):
            bot.navigate_to_bounty_contract_screen()

        open_tab.assert_called_once_with(
            card_template="battle_modes/bounty_contract.png"
        )
        assert tap_till.call_args_list == [
            call("battle_modes/bounty_contract.png"),
            call("bounty_contract/hall_label.png", threshold=ConfidenceValue("80%")),
        ]
        assert _SEARCH in wait_any.call_args.kwargs["templates"]


class TestOpenGuildModeTab:
    def test_skips_tab_when_card_already_visible(self):
        bot = _Stub()
        with (
            patch.object(bot, "game_find_template_match", return_value=MagicMock()),
            patch.object(bot, "tap") as tap,
        ):
            bot._open_guild_mode_tab(card_template="battle_modes/bounty_contract.png")

        tap.assert_not_called()

    def test_taps_tab_when_card_not_visible(self):
        bot = _Stub()
        tab = MagicMock()
        with (
            patch.object(bot, "game_find_template_match", return_value=None),
            patch.object(bot, "wait_for_template", return_value=tab),
            patch.object(bot, "tap") as tap,
            patch.object(bot, "sleep_navigation"),
        ):
            bot._open_guild_mode_tab(card_template="battle_modes/union_campaign.png")

        tap.assert_called_once_with(tab)


_DATA_DIR = Path(__file__).parents[1] / "data"
_TEMPLATE_DIR = (
    Path(__file__).parents[4] / "adb_auto_player/games/afk_journey/templates"
)
# bounty_contract_bottom.png is the bottom strip (y >= 85%) of a 1080x1920
# capture, i.e. exactly the region BOUNTY_CONTRACT_BOTTOM_CROP keeps.
_STRIP_TOP = 1632


def _imread(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read {path}"
    return image


class TestOnRealScreen:
    def test_crop_matches_test_strip(self):
        assert CropRegions(top=0.85) == BountyContractMixin.BOUNTY_CONTRACT_BOTTOM_CROP

    def test_search_targets_template_matches(self):
        strip = _imread(_DATA_DIR / "bounty_contract_bottom.png")
        assert TemplateMatcher.find_template_match(
            strip, _imread(_TEMPLATE_DIR / _SEARCH)
        )
        assert (
            TemplateMatcher.find_template_match(
                strip, _imread(_TEMPLATE_DIR / _DEPLETED)
            )
            is None
        )

    def test_depleted_template_matches_grey_button_only(self):
        strip = _imread(_DATA_DIR / "bounty_contract_depleted_bottom.png")
        assert TemplateMatcher.find_template_match(
            strip, _imread(_TEMPLATE_DIR / _DEPLETED)
        )
        assert (
            TemplateMatcher.find_template_match(strip, _imread(_TEMPLATE_DIR / _SEARCH))
            is None
        )

    def test_counter_ocr_reads_remaining_searches(self):
        strip = _imread(_DATA_DIR / "bounty_contract_bottom.png")
        screenshot = np.zeros((1920, 1080, 3), dtype=np.uint8)
        screenshot[_STRIP_TOP:] = strip

        assert _Stub()._read_search_targets_counter(screenshot) == 15
