"""Tests for the Dailies hamburger-menu claims."""

from unittest.mock import patch

import pytest
from adb_auto_player.games.afk_journey.mixins.dailies import DailiesMixin


class _Stub(DailiesMixin):
    def __init__(self) -> None:
        pass

    @property
    def template_timeout(self) -> float:
        return 30.0


@pytest.mark.parametrize(
    ("method", "optional_template"),
    [
        ("_claim_mail", "dailies/hamburger/read_all.png"),
        ("_claim_friend_rewards", "dailies/hamburger/send_receive.png"),
    ],
)
def test_optional_claim_button_uses_the_short_timeout(method, optional_template):
    """Regression test: a missing optional button must not cost the full timeout.

    With a 30s template timeout, "No mail" waited 30s for Read All.
    """
    bot = _Stub()
    timeouts: dict[str, float | None] = {}

    def _try_wait_and_tap(template, timeout=None, **_):
        timeouts[template] = timeout
        return template != optional_template

    with (
        patch.object(bot, "_try_wait_and_tap", side_effect=_try_wait_and_tap),
        patch.object(bot, "game_find_template_match", return_value=None),
        patch.object(bot, "press_back_button"),
        patch.object(bot, "tap"),
    ):
        getattr(bot, method)()

    assert timeouts[optional_template] == bot.fast_timeout
    assert bot.fast_timeout < bot.template_timeout
