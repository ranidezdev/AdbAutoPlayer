"""Tests for the Supreme Arena battle flow."""

from unittest.mock import MagicMock, patch

from adb_auto_player.exceptions import GameTimeoutError
from adb_auto_player.games.afk_journey.mixins.supreme_arena import SupremeArenaMixin
from adb_auto_player.games.afk_journey.popup_message_handler import (
    PopupMessageHandler,
)


class _Stub(SupremeArenaMixin):
    def __init__(self) -> None:
        pass

    @property
    def template_timeout(self) -> float:
        return 1.0

    def sleep_action(self) -> None:
        pass

    def sleep_navigation(self) -> None:
        pass


def test_no_artifact_popup_is_a_known_confirm_popup():
    popup = PopupMessageHandler._find_matching_popup(
        "No Artifact equipped. Continue to start the battle?"
    )

    assert popup is not None
    assert not popup.ignore
    assert popup.confirm_button_template == "navigation/confirm.png"


def test_popups_are_handled_before_waiting_for_the_battle_to_end():
    """Regression test: a start-battle popup must be confirmed first.

    Previously the popup was only checked after waiting for the Skip button,
    and the battle-over wait then tapped the popup's Confirm button, treating
    a battle that had not even started as finished.
    """
    bot = _Stub()
    calls: list[str] = []

    def _wait_for_template(template, **_):
        calls.append(f"wait:{template}")
        if template == "arena/skip.png":
            raise GameTimeoutError("no skip")
        return MagicMock(template=template)

    with (
        patch.object(bot, "wait_for_template", side_effect=_wait_for_template),
        patch.object(
            bot,
            "wait_for_any_template",
            side_effect=GameTimeoutError("no popup after battle"),
        ),
        patch.object(
            bot, "tap", side_effect=lambda match: calls.append(f"tap:{match.template}")
        ),
        patch.object(
            bot,
            "handle_popup_messages",
            side_effect=lambda: calls.append("popups") or False,
        ),
    ):
        bot._sa_battle()

    battle_tap = calls.index("tap:arena/battle.png")
    assert calls[battle_tap + 1] == "popups"
    assert calls.index("popups") < calls.index("wait:arena/skip.png")
