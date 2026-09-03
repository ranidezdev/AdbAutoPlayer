"""Regression tests for the Primal Lord battle loop stop conditions."""

from unittest.mock import MagicMock, patch

from adb_auto_player.exceptions import GameTimeoutError
from adb_auto_player.games.afk_journey.mixins.primal_lord import PrimalLordMixin


class _Stub(PrimalLordMixin):
    """Minimal stub exercising only the pure battle-loop control flow."""

    min_timeout = 1.0
    fast_timeout = 1.0

    def __init__(self) -> None:
        pass


def test_run_battles_stops_when_boss_battle_button_missing() -> None:
    bot = _Stub()
    with (
        patch.object(bot, "_try_wait_and_tap", return_value=False),
        patch.object(bot, "wait_for_template") as wait_for_template,
        patch.object(bot, "sleep_navigation"),
        patch.object(bot, "_wait_for_battle_end") as wait_for_battle_end,
    ):
        bot._run_battles()

    wait_for_battle_end.assert_not_called()
    wait_for_template.assert_not_called()


def test_run_battles_stops_when_formation_screen_not_found() -> None:
    bot = _Stub()
    with (
        patch.object(bot, "_try_wait_and_tap", return_value=True),
        patch.object(
            bot, "wait_for_template", side_effect=GameTimeoutError("no formation")
        ),
        patch.object(bot, "sleep_navigation"),
        patch.object(bot, "press_back_button") as press_back_button,
        patch.object(bot, "_wait_for_battle_end") as wait_for_battle_end,
    ):
        bot._run_battles()

    press_back_button.assert_called_once()
    wait_for_battle_end.assert_not_called()


def test_run_battles_runs_up_to_max_attempts() -> None:
    bot = _Stub()
    with (
        patch.object(bot, "_try_wait_and_tap", return_value=True),
        patch.object(bot, "wait_for_template", return_value=MagicMock()),
        patch.object(bot, "sleep_navigation"),
        patch.object(bot, "_wait_for_battle_end") as wait_for_battle_end,
    ):
        bot._run_battles()

    assert wait_for_battle_end.call_count == PrimalLordMixin._MAX_ATTEMPTS
