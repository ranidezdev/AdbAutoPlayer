"""Tests for the shared Showdown event flow (Sunlit / Frostfire)."""

from unittest.mock import MagicMock, patch

from adb_auto_player.games.afk_journey.mixins.frostfire_showdown import (
    FrostfireShowdownMixin,
)
from adb_auto_player.games.afk_journey.mixins.sunlit_showdown import SunlitShowdownMixin

_SLEEP = "adb_auto_player.games.afk_journey.mixins._showdown_event.sleep"


class _Sunlit(SunlitShowdownMixin):
    def __init__(self) -> None:
        pass


class _Frostfire(FrostfireShowdownMixin):
    def __init__(self) -> None:
        self._max_stamina = 3
        self._max_retries = 3
        self._difficulty_name = "Unknown"


class TestStartNextRun:
    def test_out_of_tokens_returns_false(self):
        bot = _Sunlit()

        with (
            patch.object(bot, "wait_for_template") as mock_wait,
            patch.object(bot, "tap") as mock_tap,
            patch.object(bot, "game_find_template_match", return_value=MagicMock()),
            patch(_SLEEP),
        ):
            assert bot._start_next_run() is False

        mock_wait.assert_called_once_with(template="event/sunlit_showdown/title_s")
        mock_tap.assert_called_once_with(bot._START_POINT)

    def test_tokens_left_returns_true(self):
        bot = _Sunlit()

        with (
            patch.object(bot, "wait_for_template"),
            patch.object(bot, "tap"),
            patch.object(
                bot, "game_find_template_match", return_value=None
            ) as mock_find,
            patch(_SLEEP),
        ):
            assert bot._start_next_run() is True

        mock_find.assert_called_once_with(
            template="event/sunlit_showdown/insufficient_resources"
        )

    def test_frostfire_detects_difficulty_before_starting(self):
        bot = _Frostfire()
        calls: list[str] = []

        with (
            patch.object(bot, "wait_for_template"),
            patch.object(
                bot,
                "_detect_difficulty",
                side_effect=lambda: calls.append("detect") or ("Epic", 2, 2),
            ),
            patch.object(bot, "tap", side_effect=lambda *_: calls.append("tap")),
            patch.object(bot, "game_find_template_match", return_value=None),
            patch(_SLEEP),
            patch("adb_auto_player.games.afk_journey.mixins.frostfire_showdown.sleep"),
        ):
            assert bot._start_next_run() is True

        assert calls == ["detect", "tap"]
        assert (bot._difficulty_name, bot._max_stamina, bot._max_retries) == (
            "Epic",
            2,
            2,
        )


class TestOpenShowdownEvent:
    def test_navigates_to_the_event_entry(self):
        bot = _Frostfire()

        with (
            patch.object(bot, "_navigate_menu_chain") as mock_chain,
            patch.object(bot, "wait_for_template"),
            patch.object(bot, "tap"),
            patch.object(bot, "_start_next_run", return_value=True) as mock_start,
            patch(_SLEEP),
        ):
            assert bot._open_frostfire_showdown() is True

        mock_chain.assert_called_once_with(
            [
                "navigation/hamburger_menu",
                "dailies/hamburger/events",
                "event/frostfire_showdown/frostfire_showdown",
            ]
        )
        mock_start.assert_called_once()
