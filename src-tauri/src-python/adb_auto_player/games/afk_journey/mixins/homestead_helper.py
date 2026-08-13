"""Homestead helper mixin."""

import logging
import re
from enum import Enum
from itertools import pairwise
from time import sleep
from typing import ClassVar

import cv2
import numpy as np
from adb_auto_player.decorators import register_command, register_custom_routine_choice
from adb_auto_player.exceptions import (
    AutoPlayerUnrecoverableError,
    AutoPlayerWarningError,
    GameTimeoutError,
)
from adb_auto_player.games.afk_journey.base import AFKJourneyBase
from adb_auto_player.games.afk_journey.gui_category import AFKJCategory
from adb_auto_player.models import ConfidenceValue
from adb_auto_player.models.decorators import GUIMetadata
from adb_auto_player.models.geometry import Point
from adb_auto_player.models.image_manipulation import CropRegions
from adb_auto_player.models.template_matching import TemplateMatchResult
from adb_auto_player.ocr import RapidOCRBackend
from adb_auto_player.util import SummaryGenerator


class _RequestFulfillment(Enum):
    """Outcome of attempting to fulfill a single homestead request."""

    DELIVERED = "delivered"
    CRAFTED = "crafted"
    NOTHING = "nothing"


class HomesteadHelperMixin(AFKJourneyBase):
    """Homestead helper mixin."""

    # Templates - resource collection
    HOMESTEAD_OVERVIEW_CHECK_TEMPLATE = "homestead/homestead_overview_check.png"
    HOMESTEAD_BUILDINGS_TAB_TEMPLATE = "homestead/buildings_tab.png"
    HOMESTEAD_MINE_TEMPLATE = "homestead/mine_building.png"
    HOMESTEAD_MINE_GO_TEMPLATE = "homestead/mine_go_button.png"
    HOMESTEAD_HARVEST_ALL_TEMPLATE = "homestead/harvest_all.png"

    # Templates - orders / requests
    HOMESTEAD_REQUESTS_TEMPLATE = "homestead/requests_label.png"
    HOMESTEAD_QUICK_SELECT_TEMPLATE = "homestead/quick_select.png"
    HOMESTEAD_DELIVER_TEMPLATE = "homestead/deliver_button.png"
    HOMESTEAD_ORDER_COMPLETE_TEMPLATE = "homestead/order_complete.png"
    HOMESTEAD_MISSING_RESOURCES_TEMPLATE = (
        "homestead/missing_item_navigate_to_crafting.png"
    )

    # Templates - crafting multiplier & action
    HOMESTEAD_MULTIPLIER_X10_TEMPLATE = "homestead/multiplier_x10.png"
    HOMESTEAD_MULTIPLIER_STATE_TEMPLATE = "homestead/multiplier_state.png"
    HOMESTEAD_CRAFTING_SCREEN_TEMPLATE = "homestead/crafting_screen_check.png"
    # "Deck Setup" text label, only present on the crafting screen. Together
    # with the dish/card row it is a craft-type-agnostic anchor that (unlike the
    # colour-agnostic multiplier/action buttons) never false-matches on the
    # transition screen while the player walks to the crafting building.
    HOMESTEAD_DECK_SETUP_TEMPLATE = "homestead/deck_in_crafting_page.png"
    HOMESTEAD_ACTION_BUTTON_TEMPLATES: ClassVar[tuple[str, ...]] = (
        "homestead/cook_button.png",
        "homestead/alchem_button.png",
        "homestead/forge_button.png",
    )

    # Templates - greyed-out action button (a required ingredient is missing and
    # can itself be crafted). Appears in the centre action-button slot.
    HOMESTEAD_GRAY_ACTION_BUTTON_TEMPLATES: ClassVar[tuple[str, ...]] = (
        "homestead/gray_make_button.png",
        "homestead/gray_alchem_button.png",
        "homestead/gray_forge_button.png",
    )

    # Templates - ingredient crafting screen (reached after tapping the grey
    # action button and following the missing-ingredient popup arrow).
    HOMESTEAD_INGREDIENT_ACTION_BUTTON_TEMPLATES: ClassVar[tuple[str, ...]] = (
        "homestead/ingredient_smelt_button.png",
        "homestead/ingredient_shape_button.png",
        "homestead/ingredient_refine_button.png",
    )
    HOMESTEAD_INGREDIENT_TAP_TO_CLOSE_TEMPLATE = "homestead/ingredient_tap_to_close.png"

    # Templates - "Process Cards available to upgrade" popup + upgrade screen.
    # The popup can appear on the way to the crafting screen; its green check
    # accepts it and leads to the Process Upgrade screen where the upgradeable
    # card (flagged with a small green arrow) is pre-selected, so pressing the
    # green Upgrade button upgrades it.
    HOMESTEAD_PROCESS_UPGRADE_POPUP_TEMPLATE = "homestead/process_upgrade_confirm.png"
    HOMESTEAD_PROCESS_UPGRADE_BUTTON_TEMPLATE = "homestead/process_upgrade_button.png"

    # Template - the "Stamina Bundle" shop popup that appears when Homestead
    # stamina is depleted and a harvest/craft action is attempted. Its presence
    # means no stamina is left, so the mode should conclude.
    HOMESTEAD_NO_STAMINA_TEMPLATE = "homestead/stamina_bundle_popup.png"

    # Fixed tap point for the multiplier button (cycles x1 -> x5 -> x10)
    HOMESTEAD_MULTIPLIER_BUTTON_POINT = Point(760, 1660)
    # Fixed tap point for the action button (Make / Alchemize / Forge)
    HOMESTEAD_ACTION_BUTTON_POINT = Point(540, 1680)
    # Fixed tap point for the greyed-out action button (centre slot).
    HOMESTEAD_GRAY_ACTION_BUTTON_POINT = Point(520, 1675)
    # Fixed tap point for the ingredient crafting action button (Smelt/Shape/...).
    HOMESTEAD_INGREDIENT_ACTION_BUTTON_POINT = Point(631, 1811)
    # Tap point to dismiss the "Tap to close" rewards popup (bottom of screen).
    HOMESTEAD_TAP_TO_CLOSE_POINT = Point(540, 1800)
    # Fixed tap point for the Requests world-icon (always at same position)
    HOMESTEAD_REQUESTS_POINT = Point(60, 1245)

    # The four request NPC portraits at the bottom of the Requests view. Tapping
    # one selects that request and updates the "Basic Rewards" panel at the top.
    HOMESTEAD_REQUEST_PORTRAIT_POINTS: ClassVar[tuple[Point, ...]] = (
        Point(270, 1815),
        Point(452, 1815),
        Point(640, 1815),
        Point(822, 1815),
    )
    # Crop region (x1, y1, x2, y2) of the "Basic Rewards" panel for the
    # currently selected request. It deliberately spans BOTH reward cards:
    # the Wish Points number (left card) and the Ancient Coins number (right
    # card). We OCR the whole band, keep only numeric blocks and pick the
    # left-most one as the Wish Points value. A wide crop avoids clipping
    # wider (4-5 digit) numbers, and using the left-most block guarantees we
    # never accidentally read the Ancient Coins number on the right.
    # The number's vertical position varies (~y=768 for most requests, ~y=819
    # for the request pre-selected when the view opens), so the crop is tall
    # enough to cover both.
    HOMESTEAD_REWARDS_CROP: ClassVar[tuple[int, int, int, int]] = (
        150,
        745,
        560,
        850,
    )
    # After selecting a portrait the Basic Rewards panel animates in, so the
    # number is briefly blank. Retry the OCR read a few times until it appears.
    HOMESTEAD_WISH_POINT_READ_ATTEMPTS = 5
    HOMESTEAD_WISH_POINT_READ_DELAY = 0.6
    # Maximum horizontal gap (px, in crop space) between two numeric OCR blocks
    # for them to be treated as parts of the same reward number. OCR sometimes
    # splits a single value (e.g. "9300" -> "93" + "00"); such fragments sit
    # right next to each other, whereas the Ancient Coins number on the far
    # right is separated by a much larger gap and forms its own group.
    HOMESTEAD_CARD_GAP_MAX = 80

    # Ingredient-crafting slider geometry. The handle starts at the far left
    # (value 0); dragging right increases the craft amount. The track spans
    # roughly x=250..900. We pull ~20% of the way to queue a small batch.
    HOMESTEAD_SLIDER_Y = 1582
    HOMESTEAD_SLIDER_START_X = 245
    HOMESTEAD_SLIDER_END_X = 375

    # Tuning
    # Total attempts for the whole cycle: one initial run plus one retry.
    HOMESTEAD_CYCLE_ATTEMPTS = 2
    HOMESTEAD_MINE_SCROLL_ATTEMPTS = 5
    HOMESTEAD_HARVEST_TIMEOUT = 30
    HOMESTEAD_OUTER_LOOP_LIMIT = 15
    HOMESTEAD_INNER_LOOP_LIMIT = 25
    # Abort ingredient crafting if nothing progresses within this many seconds.
    HOMESTEAD_INGREDIENT_CRAFT_TIMEOUT = 30
    # Attempts (each followed by 0.5s) to detect the insufficient-resources popup
    # arrow after tapping a craft/action button before assuming crafting started.
    HOMESTEAD_ARROW_DETECT_ATTEMPTS = 6
    # Attempts (each followed by 0.5s) to confirm the crafting screen is present
    # and stable after navigation before interacting with the multiplier button.
    # The player walks to the crafting building first, so the screen only appears
    # after a delay; require it on two consecutive checks to avoid tapping the
    # multiplier on the transition screen.
    HOMESTEAD_CRAFTING_CONFIRM_ATTEMPTS = 12
    # Consecutive positive checks required before the crafting screen is
    # considered stable enough to interact with.
    HOMESTEAD_CRAFTING_CONFIRM_STREAK = 2
    # A ready action button is coloured; a disabled (ingredient-missing) button
    # is rendered as a pure-grey circle. Mean HSV saturation below this value
    # means the button is greyed out. The action-button templates match the
    # coloured and grey states at nearly the same confidence, so colour is the
    # only reliable way to tell them apart.
    HOMESTEAD_DISABLED_BUTTON_SATURATION_MAX = 25

    @register_command(
        name="HomesteadOrdersHelper",
        gui=GUIMetadata(
            label="Homestead Orders Helper",
            category=AFKJCategory.GAME_MODES,
            tooltip="Collect Mine resources and fulfill Requests orders in Homestead",
        ),
    )
    @register_custom_routine_choice(label="Homestead Orders Helper")
    def homestead_orders_helper(self) -> None:
        """Collect Mine resources and fulfill Homestead Requests orders."""
        self.start_up()
        try:
            self._run_homestead_cycle_with_retry()
        finally:
            # Leave Homestead so any task that runs next (even after a
            # failure here) starts from World - otherwise its Battle-Modes-
            # style navigation could mistap a Homestead building instead of
            # the intended World button.
            self.navigate_to_world()

    def _run_homestead_cycle_with_retry(self) -> None:
        """Run the full homestead cycle, retrying once on any recoverable error.

        Any unexpected exception re-runs the whole cycle exactly one more time.
        If the retry also fails the error is re-raised so the mode quits.
        Unrecoverable errors are never retried.
        """
        for attempt in range(1, self.HOMESTEAD_CYCLE_ATTEMPTS + 1):
            try:
                self._run_homestead_cycle()
                return
            except AutoPlayerUnrecoverableError:
                raise
            except AutoPlayerWarningError:
                # e.g. no stamina left - conclude the mode without retrying.
                raise
            except Exception:
                if attempt >= self.HOMESTEAD_CYCLE_ATTEMPTS:
                    logging.exception(
                        "Homestead cycle failed again on retry - quitting mode."
                    )
                    raise
                logging.exception(
                    "Homestead cycle hit an unexpected error - "
                    "retrying the whole cycle once."
                )

    def _conclude_if_no_stamina(self) -> None:
        """End the mode cleanly if the out-of-stamina popup is showing.

        Homestead harvesting and crafting consume stamina. When it runs out the
        game shows a "Stamina Bundle" shop popup instead of performing the
        action. Dismiss it and raise so the mode concludes without retrying.

        Raises:
            AutoPlayerWarningError: When the Stamina Bundle popup is detected.
        """
        if (
            self.game_find_template_match(
                template=self.HOMESTEAD_NO_STAMINA_TEMPLATE,
                threshold=ConfidenceValue("80%"),
            )
            is None
        ):
            return
        logging.info("Homestead stamina depleted - closing popup and ending mode.")
        self.tap(self.HOMESTEAD_TAP_TO_CLOSE_POINT)
        sleep(1)
        raise AutoPlayerWarningError(
            "No Homestead stamina remaining - Homestead Orders Helper finished."
        )

    def _run_homestead_cycle(self) -> None:
        """Run one full homestead cycle.

        Enters homestead, collects Mine resources and fulfills Requests orders.
        """
        self._ensure_in_homestead()
        self._collect_homestead_resources()
        self._handle_homestead_requests()

    def _ensure_in_homestead(self) -> None:
        """Enter homestead if not already there.

        Uses the stacked-coins icon (top-right) as the definitive signal that
        we are inside homestead. Presses back to escape any open sub-screens
        (crafting, requests, etc.) before trying to enter.
        Raises GameTimeoutError after 10 attempts.
        """
        for attempt in range(10):
            # Already in homestead world view?
            if self.game_find_template_match(
                template=self.HOMESTEAD_OVERVIEW_CHECK_TEMPLATE
            ):
                logging.info("Already in homestead.")
                return

            # Look for the green Homestead enter button (world view, not inside).
            enter = self.find_any_template(
                [
                    "navigation/homestead/homestead_enter.png",
                    "navigation/homestead/homestead_invaded.png",
                ]
            )
            if enter is not None:
                logging.info(
                    "Tapping homestead enter button (attempt %d).", attempt + 1
                )
                self.tap(enter)
                sleep(4)  # wait for homestead to load
                continue

            # We are inside a sub-screen (crafting, requests, etc.).
            # Press back to get closer to the homestead world view.
            # If the exit-homestead dialog appears, dismiss it with Cancel.
            logging.debug(
                "Sub-screen detected (attempt %d) - pressing back.", attempt + 1
            )
            self.press_back_button()
            sleep(2)
            cancel = self.game_find_template_match(template="cancel.png")
            if cancel is not None:
                logging.info("Exit dialog detected - dismissing with Cancel.")
                self.tap(cancel)
                sleep(2)

        raise GameTimeoutError("Could not navigate to homestead after 10 attempts.")

    # ------------------------------------------------------------------ #
    #  Resource collection                                                 #
    # ------------------------------------------------------------------ #

    def _collect_homestead_resources(self) -> None:
        """Open Buildings -> Mine -> Go -> Harvest All."""
        logging.info("Collecting homestead resources...")

        # Open the management panel and navigate to the Buildings tab.
        # The panel may already be open after navigation, so we try to find
        # the Buildings tab first.  If it isn't visible, tap the stacked-coins
        # icon to toggle the panel open and try again (up to 3 times).
        sleep(2)  # let the UI settle after navigate_to_homestead
        buildings_tab = None
        for attempt in range(3):
            buildings_tab = self.game_find_template_match(
                template=self.HOMESTEAD_BUILDINGS_TAB_TEMPLATE
            )
            if buildings_tab is not None:
                break
            # Panel not open yet - find and tap the stacked-coins icon.
            overview_icon = self.game_find_template_match(
                template=self.HOMESTEAD_OVERVIEW_CHECK_TEMPLATE
            )
            if overview_icon is None:
                logging.warning(
                    "Neither Buildings tab nor overview icon found (attempt %d).",
                    attempt + 1,
                )
                sleep(2)
                continue
            logging.info(
                "Tapping stacked-coins icon to open panel (attempt %d).", attempt + 1
            )
            self.tap(overview_icon)
            sleep(3)  # wait for the panel animation to finish

        if buildings_tab is None:
            logging.warning("Could not open the homestead management panel.")
            return

        self.tap(buildings_tab)
        sleep(2)

        # Scroll down until a Mine building card is visible.
        mine = None
        for _ in range(self.HOMESTEAD_MINE_SCROLL_ATTEMPTS):
            mine = self.game_find_template_match(
                template=self.HOMESTEAD_MINE_TEMPLATE,
                threshold=ConfidenceValue("75%"),
                grayscale=True,
            )
            if mine is not None:
                break
            self.swipe_up(x=540, sy=1500, ey=700)
            sleep(1.5)

        if mine is None:
            logging.warning(
                "Mine building not found after scrolling"
                " - skipping resource collection."
            )
            self.press_back_button()
            sleep(1)
            return

        self.tap(mine)
        sleep(2)

        # Press the green "Go" button to navigate to the Mine in the world.
        go_button = self.wait_for_template(
            template=self.HOMESTEAD_MINE_GO_TEMPLATE,
            timeout=self.navigation_timeout,
            timeout_message="Could not find Mine Go button.",
        )
        self.tap(go_button)
        sleep(3)

        # Wait for the Harvest All button and press it.
        # Use a lower threshold and restrict to the left half of the screen
        # where the world-space mine icon always appears.
        harvest_all = self.wait_for_template(
            template=self.HOMESTEAD_HARVEST_ALL_TEMPLATE,
            threshold=ConfidenceValue("70%"),
            grayscale=True,
            crop_regions=CropRegions(right=0.5),
            timeout=self.HOMESTEAD_HARVEST_TIMEOUT,
            timeout_message="Harvest All button not found.",
        )
        self.tap(harvest_all)
        sleep(4)  # wait for harvest animation and camera to settle

        self._conclude_if_no_stamina()

        SummaryGenerator.increment("Homestead Orders Helper", "Resources Harvested")
        logging.info("Resources harvested.")

    # ------------------------------------------------------------------ #
    #  Orders / Requests                                                   #
    # ------------------------------------------------------------------ #

    def _handle_homestead_requests(self) -> None:
        """Repeatedly open Requests, pick the best-rewarded order and fulfill it.

        Each time the Requests view is entered, the four request portraits are
        compared by their Wish Point reward and the highest one is selected
        before fulfilling. After a craft trip the view is re-entered and the
        comparison is repeated.
        """
        logging.info("Handling homestead requests...")

        for _outer in range(self.HOMESTEAD_OUTER_LOOP_LIMIT):
            # Verify Requests icon is visible before tapping fixed coordinate.
            requests_visible = self.game_find_template_match(
                template=self.HOMESTEAD_REQUESTS_TEMPLATE,
                threshold=ConfidenceValue("60%"),
                grayscale=True,
                crop_regions=CropRegions(right=0.75, top=0.55),
            )
            if requests_visible is None:
                logging.info("Requests button not visible - done.")
                return

            logging.info("Tapping Requests icon.")
            self.tap(self.HOMESTEAD_REQUESTS_POINT)
            sleep(2)

            crafted_this_round = self._fulfill_requests_best_first()

            if crafted_this_round:
                # After a crafting trip the bot may still be inside the crafting
                # or requests screen. Navigate back to the homestead world view
                # before the next outer-loop iteration checks for the Requests icon.
                self._ensure_in_homestead()
                continue

            # No missing-resource craft needed - nothing left to do.
            logging.info("No orders remaining - done.")
            self.press_back_button()
            sleep(2)
            return

        logging.info("Homestead requests: reached outer loop limit.")

    def _fulfill_requests_best_first(self) -> bool:
        """Fulfill requests best-first until a craft cycle or exhaustion.

        Each iteration compares the four request portraits by their Wish Point
        reward, selects the highest and fulfills it once. Requests that cannot
        be progressed are skipped for the remainder of this Requests visit.

        Returns:
            True  if a crafting trip was made (caller should re-enter Requests).
            False if nothing more to do in this Requests visit.
        """
        exhausted: set[int] = set()
        # Cache Wish Point values across iterations. Only the request that was
        # just delivered is replaced by a new order, so the other portraits keep
        # their previously-read values and do not need to be OCR'd again.
        cache: dict[int, int] = {}

        for _ in range(self.HOMESTEAD_INNER_LOOP_LIMIT):
            selected = self._select_best_request(exclude=exhausted, cache=cache)
            if selected is None:
                logging.info("No selectable request - no more orders.")
                return False

            result = self._fulfill_selected_request()

            if result is _RequestFulfillment.CRAFTED:
                return True  # caller will call _ensure_in_homestead() then retry
            if result is _RequestFulfillment.NOTHING:
                # This request cannot be progressed right now; skip it and try
                # the next-best one in the following iteration.
                exhausted.add(selected)
            else:
                # DELIVERED: this slot now shows a new order, so its cached value
                # is stale. Invalidate it to force a fresh read next iteration;
                # the remaining slots keep their cached values.
                cache.pop(selected, None)

        logging.info("Request fulfillment inner loop limit reached.")
        return False

    def _read_request_wish_points(
        self, exclude: set[int], cache: dict[int, int]
    ) -> dict[int, int]:
        """Tap each request portrait and OCR its Wish Point reward value.

        Portraits whose value is already present in ``cache`` are not tapped or
        re-read; their previously-read value is reused. Newly read values are
        written back into ``cache``.

        Args:
            exclude: Portrait indices to skip (already exhausted this visit).
            cache: Known Wish Point values from previous comparisons. Updated
                in place with any values read during this call.

        Returns:
            Mapping of portrait index (0-based) to Wish Point value. Portraits
            in ``exclude`` or whose number could not be read are omitted.
        """
        backend = getattr(self, "_homestead_ocr_backend", None)
        if backend is None:
            backend = RapidOCRBackend()
            self._homestead_ocr_backend = backend

        x1, y1, x2, y2 = self.HOMESTEAD_REWARDS_CROP
        for index, point in enumerate(self.HOMESTEAD_REQUEST_PORTRAIT_POINTS):
            if index in exclude or index in cache:
                continue
            self.tap(point)
            # The Basic Rewards panel animates in after selecting a portrait, so
            # the number is briefly blank. Retry until it is readable.
            value: int | None = None
            for _ in range(self.HOMESTEAD_WISH_POINT_READ_ATTEMPTS):
                sleep(self.HOMESTEAD_WISH_POINT_READ_DELAY)
                crop = self.get_screenshot()[y1:y2, x1:x2]
                value = self._read_wish_points_from_crop(backend, crop)
                if value is not None:
                    break
            if value is not None:
                cache[index] = value
                logging.debug("Request %d Wish Points: %d", index + 1, value)
            else:
                logging.debug("Request %d Wish Points unreadable.", index + 1)
        return {index: value for index, value in cache.items() if index not in exclude}

    @staticmethod
    def _read_wish_points_from_crop(
        backend: RapidOCRBackend,
        crop: np.ndarray,
    ) -> int | None:
        """Extract the Wish Point value from a Basic Rewards crop.

        The crop spans both reward cards. Wish Points is the left card and
        Ancient Coins the right card, so of all numeric text blocks the
        left-most one is the Wish Point value. OCR occasionally splits a single
        number into adjacent blocks (e.g. "9300" -> "93" + "00"), so numeric
        blocks are first grouped by horizontal proximity and each group's
        fragments are concatenated before the left-most group is returned.

        Args:
            backend: OCR backend used to detect text blocks.
            crop: Cropped image (numpy array) of the Basic Rewards band.

        Returns:
            The Wish Point value, or None if no numeric block was found.
        """
        blocks: list[tuple[int, int, str]] = []
        for result in backend.detect_text_blocks(crop):
            if re.search(r"\d", result.text):
                blocks.append((result.box.left, result.box.right, result.text))
        if not blocks:
            return None

        blocks.sort(key=lambda b: b[0])
        # Group fragments of the same number (small gap) into reward cards. The
        # left-most group is Wish Points; the far-right group is Ancient Coins.
        groups: list[list[tuple[int, int, str]]] = [[blocks[0]]]
        for prev, block in pairwise(blocks):
            if block[0] - prev[1] <= HomesteadHelperMixin.HOMESTEAD_CARD_GAP_MAX:
                groups[-1].append(block)
            else:
                groups.append([block])

        return HomesteadHelperMixin._parse_reward_group(
            [block[2] for block in groups[0]]
        )

    @staticmethod
    def _parse_reward_group(texts: list[str]) -> int | None:
        """Parse the numeric value of one reward card from its OCR fragments.

        A single, unsplit block is parsed with ``_parse_reward_number`` so
        ``k``/``m`` suffixes and thousands separators are honoured. When OCR
        splits the number across several blocks they are plain integer
        fragments (a value large enough to be split is never abbreviated), so
        their digits are concatenated in left-to-right order.

        Args:
            texts: OCR text fragments of a single reward card, ordered left to
                right.

        Returns:
            The parsed integer value, or None if no digits were found.
        """
        if len(texts) == 1:
            return HomesteadHelperMixin._parse_reward_number(texts[0])
        digits = "".join(re.sub(r"\D", "", text) for text in texts)
        return int(digits) if digits else None

    @staticmethod
    def _parse_reward_number(text: str) -> int | None:
        """Parse a reward number, honoring ``k``/``m`` magnitude suffixes.

        OCR reads values such as ``12k`` or ``1.2M``. Stripping non-digits would
        turn ``12k`` into ``12``, making it compare as smaller than ``6000``.
        This expands the suffix so ``12k`` becomes ``12000``.

        Args:
            text: The OCR text block to parse.

        Returns:
            The parsed integer value, or None if no number was found.
        """
        match = re.search(r"(\d[\d.,]*)\s*([kKmM])?", text)
        if match is None:
            return None
        raw_number = match.group(1)
        suffix = match.group(2)
        if suffix:
            # With a suffix a separator is a decimal point (e.g. 1.2k, 1,2k).
            number = float(raw_number.replace(",", "."))
            multiplier = 1_000 if suffix.lower() == "k" else 1_000_000
            return int(number * multiplier)
        # Without a suffix separators are thousands groupings (e.g. 12,000).
        digits = re.sub(r"\D", "", raw_number)
        return int(digits) if digits else None

    def _select_best_request(
        self, exclude: set[int], cache: dict[int, int]
    ) -> int | None:
        """Select the request with the highest Wish Point reward.

        Reads any request portraits whose value is not already cached, taps the
        one with the highest Wish Point value (ignoring any in ``exclude``) and
        returns its index.

        Args:
            exclude: Portrait indices to skip (already exhausted this visit).
            cache: Known Wish Point values from previous comparisons. Updated
                in place with any values read during this call.

        Returns:
            The selected portrait index, or None if no request could be read.
        """
        candidates = self._read_request_wish_points(exclude=exclude, cache=cache)
        if not candidates:
            logging.info("No selectable requests with a readable reward.")
            return None

        best_index = max(candidates, key=lambda i: candidates[i])
        logging.info(
            "Selecting request %d (Wish Points: %d).",
            best_index + 1,
            candidates[best_index],
        )
        self.tap(self.HOMESTEAD_REQUEST_PORTRAIT_POINTS[best_index])
        sleep(1.5)  # let the selection settle before Quick Select
        return best_index

    def _fulfill_selected_request(self) -> _RequestFulfillment:
        """Fulfill the currently selected request once via Quick Select.

        Taps Quick Select and then either crafts a missing resource or delivers
        the order. Exactly one Quick Select action is handled so the caller can
        re-compare request rewards afterwards.

        Returns:
            CRAFTED  if a craft trip was started (caller should re-enter).
            DELIVERED if an order was delivered.
            NOTHING  if there was nothing to do for this request.
        """
        quick_select = self.game_find_template_match(
            template=self.HOMESTEAD_QUICK_SELECT_TEMPLATE
        )
        if quick_select is None:
            logging.info("Quick Select not visible for this request.")
            return _RequestFulfillment.NOTHING

        self.tap(quick_select)
        sleep(2)

        # Check if the insufficient-resources popup appeared.
        missing_arrow = self.game_find_template_match(
            template=self.HOMESTEAD_MISSING_RESOURCES_TEMPLATE
        )
        if missing_arrow is not None:
            logging.info("Insufficient resources - navigating to crafting.")
            self.tap(missing_arrow)
            sleep(2)
            self._handle_crafting_to_max()
            return _RequestFulfillment.CRAFTED

        # No missing-resources popup: check for Deliver button.
        deliver = self.game_find_template_match(
            template=self.HOMESTEAD_DELIVER_TEMPLATE
        )
        if deliver is not None:
            logging.info("Deliver button found - tapping.")
            self.tap(deliver)
            sleep(3)  # wait for reward popup to appear
            # Tap the lower half of the screen to dismiss the reward popup.
            self.tap(Point(540, 1400))
            # Wait for Quick Select to reappear.
            try:
                self.wait_for_template(
                    template=self.HOMESTEAD_QUICK_SELECT_TEMPLATE,
                    timeout=15,
                    timeout_message="Quick Select did not reappear after delivery.",
                )
            except Exception:
                logging.warning("Quick Select did not reappear after delivery.")
            SummaryGenerator.increment("Homestead Orders Helper", "Orders Delivered")
            return _RequestFulfillment.DELIVERED

        # Nothing matched for this request.
        logging.debug("No Deliver or missing-resource popup found for this request.")
        return _RequestFulfillment.NOTHING

    # ------------------------------------------------------------------ #
    #  Crafting multiplier                                                 #
    # ------------------------------------------------------------------ #

    def _handle_crafting_to_max(self) -> None:
        """Wait for crafting screen, cycle multiplier to x10, press action button.

        If the action button is greyed out, a required ingredient is missing and
        must be crafted first. In that case we follow the missing-ingredient
        popup into the ingredient crafting screen, craft a small batch, and
        return — the caller will re-enter the original craft afterwards.

        If the crafting screen never loads the request is skipped rather than
        aborting the whole mode.
        """
        action_templates = list(self.HOMESTEAD_ACTION_BUTTON_TEMPLATES)
        gray_templates = list(self.HOMESTEAD_GRAY_ACTION_BUTTON_TEMPLATES)

        logging.info("Waiting for crafting screen to load...")
        # After the navigate-to-crafting arrow is tapped the player physically
        # walks to the crafting building, so the crafting screen only appears
        # after a short delay. Gate on crafting-screen-specific anchors ("Deck
        # Setup" text and the dish/card row) rather than the multiplier/action
        # buttons: those buttons are colour-agnostic and were false-matching on
        # the transition screen, causing the multiplier to be tapped before the
        # player arrived. A "Process Cards available to upgrade" popup can also
        # appear on the way to the crafting screen.
        crafting_anchors = [
            self.HOMESTEAD_DECK_SETUP_TEMPLATE,
            self.HOMESTEAD_CRAFTING_SCREEN_TEMPLATE,
        ]
        try:
            result = self.wait_for_any_template(
                templates=[
                    self.HOMESTEAD_PROCESS_UPGRADE_POPUP_TEMPLATE,
                    *crafting_anchors,
                ],
                threshold=ConfidenceValue("80%"),
                grayscale=True,
                timeout=30,
            )
        except GameTimeoutError:
            logging.warning("Crafting screen did not load - skipping this request.")
            return

        # The process-card upgrade popup interrupted craft navigation. Accept
        # the upgrade and bail out - the caller re-enters the Requests view.
        if result.template == self.HOMESTEAD_PROCESS_UPGRADE_POPUP_TEMPLATE:
            self._handle_process_card_upgrade(result)
            return

        # Confirm the player has actually arrived on the crafting screen and it
        # is stable before interacting - never tap the multiplier while still
        # transitioning.
        if not self._confirm_crafting_screen(crafting_anchors):
            logging.warning(
                "Crafting screen not stable after navigation - skipping request."
            )
            return

        sleep(2)  # let the UI fully settle

        # A grey action button means a required ingredient is missing but can be
        # crafted. The action-button templates cannot tell the grey and coloured
        # states apart, so classify by colour (grey == zero saturation).
        if self._action_button_is_disabled():
            logging.info(
                "Greyed-out action button detected - a required ingredient is "
                "missing. Navigating to ingredient crafting."
            )
            self._handle_missing_ingredient_craft()
            return

        # Cycle x1 -> x5 -> x10 with exactly 2 taps.
        for tap_num in range(2):
            logging.debug("Tapping multiplier button (%d/2).", tap_num + 1)
            self.tap(self.HOMESTEAD_MULTIPLIER_BUTTON_POINT)
            sleep(1.5)

        # Tap the action button (Cook / Alchemize / Forge).
        logging.info("Tapping action button.")
        self.tap(self.HOMESTEAD_ACTION_BUTTON_POINT)
        sleep(2)

        # Out of stamina: the game shows the Stamina Bundle popup instead of
        # crafting. Conclude the mode.
        self._conclude_if_no_stamina()

        # Pressing a coloured action button can still raise an "insufficient
        # resources" popup when a base ingredient is too low for the chosen
        # multiplier (e.g. enough for x1 but not x10). Detect that popup and
        # craft the missing ingredient first; the caller re-enters afterwards.
        if self._craft_missing_ingredient_via_arrow():
            logging.info(
                "Insufficient resources on craft - crafted the missing ingredient."
            )
            return

        # No popup: crafting is underway. The action button is replaced by a
        # different button while crafting; wait for it to reappear — that signals
        # crafting is complete. Once a batch is crafted a previously available
        # ingredient may run out, so the button can come back greyed-out.
        sleep(3)
        logging.info("Waiting for crafting to complete...")
        try:
            self.wait_for_any_template(
                templates=action_templates + gray_templates,
                threshold=ConfidenceValue("70%"),
                grayscale=True,
                timeout=30,
            )
        except GameTimeoutError:
            logging.warning("Crafting did not complete in time - continuing.")
            return

        SummaryGenerator.increment("Homestead Orders Helper", "Items Crafted")
        logging.info("Crafting done.")

        # Crafting consumed the last of an ingredient: the button is now grey.
        # Craft the missing ingredient before returning to the caller.
        if self._action_button_is_disabled():
            logging.info(
                "Action button greyed-out after crafting - a required "
                "ingredient ran out. Navigating to ingredient crafting."
            )
            self._handle_missing_ingredient_craft()

    def _confirm_crafting_screen(self, anchors: list[str]) -> bool:
        """Confirm the crafting screen is present and stable before interacting.

        The player walks to the crafting building after the navigate arrow is
        tapped, so the crafting screen only appears once they arrive and its UI
        briefly animates in. Require a crafting anchor to be present across two
        consecutive checks before interacting so the multiplier is never tapped
        on the transition screen.

        Args:
            anchors: Crafting-screen-specific templates to look for.

        Returns:
            True if an anchor was present on two consecutive checks.
        """
        consecutive = 0
        for _ in range(self.HOMESTEAD_CRAFTING_CONFIRM_ATTEMPTS):
            present = (
                self.find_any_template(
                    anchors,
                    threshold=ConfidenceValue("80%"),
                    grayscale=True,
                )
                is not None
            )
            if present:
                consecutive += 1
                if consecutive >= self.HOMESTEAD_CRAFTING_CONFIRM_STREAK:
                    return True
            else:
                consecutive = 0
            sleep(0.5)
        return False

    def _action_button_is_disabled(self) -> bool:
        """Return True if the crafting action button is greyed out.

        A ready action button is coloured while a disabled one (a required
        ingredient is missing) is rendered as a pure-grey circle. The colour and
        grey action-button templates match both states at nearly the same
        confidence, so the button's saturation is the only reliable signal.
        """
        x = self.HOMESTEAD_ACTION_BUTTON_POINT.x
        y = self.HOMESTEAD_ACTION_BUTTON_POINT.y
        patch = self.get_screenshot()[y - 25 : y + 25, x - 25 : x + 25]
        saturation = float(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        return saturation < self.HOMESTEAD_DISABLED_BUTTON_SATURATION_MAX

    # ------------------------------------------------------------------ #
    #  Process-card upgrade                                                 #
    # ------------------------------------------------------------------ #

    def _handle_process_card_upgrade(self, popup_check: TemplateMatchResult) -> None:
        """Accept a "Process Cards available to upgrade" popup and upgrade a card.

        Taps the green check-mark on the popup, waits for the Process Upgrade
        screen and presses Upgrade for the pre-selected card. The game
        auto-selects the upgradeable card (the one flagged with the small green
        arrow badge in its top-right corner), so pressing Upgrade upgrades it.
        Returning to the Requests view is handled by the caller.

        Args:
            popup_check: The matched green check-mark button on the popup.
        """
        logging.info("Process card upgrade available - accepting.")
        self.tap(popup_check)
        sleep(2)

        try:
            upgrade_button = self.wait_for_template(
                template=self.HOMESTEAD_PROCESS_UPGRADE_BUTTON_TEMPLATE,
                timeout=self.navigation_timeout,
                timeout_message="Process Upgrade screen did not load.",
            )
        except GameTimeoutError:
            logging.warning("Process Upgrade screen did not load - skipping upgrade.")
            return

        logging.info("Upgrading process card.")
        self.tap(upgrade_button)
        sleep(3)  # wait for the upgrade animation to finish

        SummaryGenerator.increment("Homestead Orders Helper", "Process Cards Upgraded")

        # Leave the Process Upgrade screen so the caller can return to Requests.
        self.press_back_button()
        sleep(2)

    # ------------------------------------------------------------------ #
    #  Missing-ingredient crafting                                         #
    # ------------------------------------------------------------------ #

    def _handle_missing_ingredient_craft(self) -> None:
        """Craft a missing ingredient via the grey action-button sub-flow.

        Flow:
            1. Tap the grey action button -> a popup like the missing-resource
               one appears; tap its arrow to navigate to ingredient crafting.
            2. On the ingredient crafting screen, drag the amount slider ~20%
               to the right and press the green action button (Smelt/Shape/...).
            3. A "Tap to close" rewards popup appears; tap the bottom of the
               screen to dismiss it.
            4. Press back to return to the homestead world view.

        If nothing progresses within ``HOMESTEAD_INGREDIENT_CRAFT_TIMEOUT``
        seconds the sub-flow is aborted and we simply press back so the caller
        can recover.
        """
        # Tap the grey action button to open the missing-ingredient popup.
        logging.info("Tapping greyed-out action button.")
        self.tap(self.HOMESTEAD_GRAY_ACTION_BUTTON_POINT)
        sleep(2)

        if not self._craft_missing_ingredient_via_arrow():
            logging.warning(
                "Missing-ingredient popup arrow not found - aborting ingredient craft."
            )
            self.press_back_button()
            sleep(2)

    def _craft_missing_ingredient_via_arrow(self) -> bool:
        """Follow the missing-resource popup arrow and craft one ingredient batch.

        The same "insufficient resources" popup (with a navigate arrow) appears
        both when a grey action button is tapped and when a coloured action
        button is pressed without enough of a base ingredient for the chosen
        multiplier. This detects that popup and, if present, navigates to the
        ingredient crafting screen, crafts a small batch and returns to the
        homestead world view.

        Returns:
            True if the popup arrow was found and the ingredient craft sub-flow
            ran (or was aborted after navigating); False if no missing-resource
            popup was present, i.e. nothing needed crafting.
        """
        # Wait briefly for the popup arrow to animate in before giving up.
        arrow = None
        for _ in range(self.HOMESTEAD_ARROW_DETECT_ATTEMPTS):
            arrow = self.game_find_template_match(
                template=self.HOMESTEAD_MISSING_RESOURCES_TEMPLATE
            )
            if arrow is not None:
                break
            sleep(0.5)
        if arrow is None:
            return False

        logging.info("Tapping popup arrow to navigate to ingredient crafting.")
        self.tap(arrow)

        # Wait for the ingredient crafting screen (its green action button).
        ingredient_buttons = list(self.HOMESTEAD_INGREDIENT_ACTION_BUTTON_TEMPLATES)
        try:
            self.wait_for_any_template(
                templates=ingredient_buttons,
                timeout=self.HOMESTEAD_INGREDIENT_CRAFT_TIMEOUT,
                timeout_message="Ingredient crafting screen did not load.",
            )
        except GameTimeoutError:
            logging.warning(
                "Ingredient crafting screen did not load within %ds - aborting.",
                self.HOMESTEAD_INGREDIENT_CRAFT_TIMEOUT,
            )
            self.press_back_button()
            sleep(2)
            return True

        sleep(1.5)  # let the screen settle before grabbing the slider

        # Drag the amount slider ~20% to the right (from 0).
        logging.info("Dragging amount slider to the right.")
        self.device.swipe(
            Point(self.HOMESTEAD_SLIDER_START_X, self.HOMESTEAD_SLIDER_Y),
            Point(self.HOMESTEAD_SLIDER_END_X, self.HOMESTEAD_SLIDER_Y),
            duration=0.6,
        )
        sleep(1.5)

        # Press the green action button (Smelt / Shape / Refine).
        logging.info("Pressing ingredient craft button.")
        self.tap(self.HOMESTEAD_INGREDIENT_ACTION_BUTTON_POINT)

        # Out of stamina: the game shows the Stamina Bundle popup instead of
        # crafting. Conclude the mode.
        self._conclude_if_no_stamina()

        # Wait for the "Tap to close" rewards popup, then dismiss it.
        try:
            self.wait_for_template(
                template=self.HOMESTEAD_INGREDIENT_TAP_TO_CLOSE_TEMPLATE,
                threshold=ConfidenceValue("60%"),
                grayscale=True,
                timeout=self.HOMESTEAD_INGREDIENT_CRAFT_TIMEOUT,
                timeout_message="Ingredient craft did not complete.",
            )
        except GameTimeoutError:
            logging.warning(
                "Ingredient craft did not complete within %ds - aborting.",
                self.HOMESTEAD_INGREDIENT_CRAFT_TIMEOUT,
            )
            self.press_back_button()
            sleep(2)
            return True

        logging.info("Dismissing 'Tap to close' popup.")
        self.tap(self.HOMESTEAD_TAP_TO_CLOSE_POINT)
        sleep(2)

        SummaryGenerator.increment("Homestead Orders Helper", "Ingredients Crafted")

        # Navigate back to the homestead world view.
        logging.info("Returning from ingredient crafting.")
        self.press_back_button()
        sleep(2)
        return True
