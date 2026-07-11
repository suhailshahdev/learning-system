"""Exploration spike for claude.ai DOM structure.

The goal is to answer the open questions M3 production code will need:
selectors for the message input, the submit mechanism, the response
container, the response-done signal, the message count indicator if
any, and the new-chat action.

This script is throwaway. It launches real Chrome with a persistent
profile, navigates to claude.ai, sends one hardcoded message, prints
what it found, and leaves the browser open briefly for manual
inspection. Run from backend/ with:

    uv run python scripts/spike_claude_dom.py

First run: claude.ai will be logged out. The script pauses and waits
for you to log in manually. The persistent profile keeps the session
for subsequent runs.
"""

from __future__ import annotations

import time

from app.core.config import get_settings
from patchright.sync_api import Page, sync_playwright
from patchright.sync_api import TimeoutError as PlaywrightTimeout

PROFILE_PATH = get_settings().chrome_profile_path
CLAUDE_URL = "https://claude.ai"
TEST_MESSAGE = (
    "Hello, I'm testing automation. Please respond with just the word "
    "'acknowledged' and nothing else."
)
LOGIN_WAIT_SECONDS = 300
INSPECTION_WAIT_SECONDS = 30
RESPONSE_STABLE_TICKS = 3


def is_logged_in(page: Page) -> bool:
    """Heuristic: logged-in URLs do not contain '/login'.

    Worth verifying empirically; the spike's whole point is checking
    assumptions like this against reality.
    """
    return "/login" not in page.url


def wait_for_login(page: Page) -> None:
    """Pause until the user finishes logging in manually.

    Uses Playwright's url-pattern wait rather than polling page.url,
    because claude.ai is a single-page app and the polling read can
    lag the actual navigation state.
    """
    print(f"\nNot logged in. URL is {page.url}")
    print(f"Log in through the browser window. Waiting up to {LOGIN_WAIT_SECONDS}s.")
    try:
        page.wait_for_url(
            lambda url: "/login" not in url,
            timeout=LOGIN_WAIT_SECONDS * 1000,
        )
        print(f"Login detected. URL is now {page.url}")
    except PlaywrightTimeout as e:
        raise RuntimeError("Login wait timed out. Re-run the spike.") from e


def find_message_input(page: Page) -> str:
    """Try a list of plausible selectors and return the first that matches."""
    candidates = [
        'div[contenteditable="true"]',
        'textarea[placeholder*="reply" i]',
        'textarea[placeholder*="message" i]',
        '[role="textbox"]',
    ]
    for selector in candidates:
        if page.locator(selector).count() > 0:
            print(f"Message input selector that matched: {selector}")
            return selector
    raise RuntimeError("No message input found. DOM has shifted; update candidates list.")


def send_message(page: Page, selector: str, text: str) -> None:
    """Fill the input and submit. Tries Enter first; falls back to button."""
    input_box = page.locator(selector).first
    input_box.click()
    input_box.fill(text)
    print(f"Filled input with: {text!r}")

    page.keyboard.press("Enter")
    print(
        "Pressed Enter to submit. If Enter does not submit, the script "
        "will need a send-button fallback."
    )


def wait_for_response(page: Page) -> str:
    """Wait for Claude's response and return its text.

    The strategy is generic: look for a stable response container, then
    wait until its text stops changing for a short window (a streaming
    completion signal). We refine this once we see the actual DOM.
    """
    response_selectors = [
        '[data-testid*="message"]',
        '[class*="message"]',
        "article",
    ]
    last_text = ""
    stable_count = 0
    deadline = time.time() + 60

    while time.time() < deadline:
        for selector in response_selectors:
            elements = page.locator(selector).all()
            if not elements:
                continue
            current_text = elements[-1].inner_text()
            if current_text and current_text != last_text:
                last_text = current_text
                stable_count = 0
            elif current_text == last_text and current_text:
                stable_count += 1
            if stable_count >= RESPONSE_STABLE_TICKS and last_text:
                print(f"Response stabilized using selector: {selector}")
                return last_text
        time.sleep(1)

    if last_text:
        print("Response did not fully stabilize within 60s. Returning last seen.")
        return last_text
    raise RuntimeError("No response detected within 60 seconds.")


def report_findings(page: Page, response_text: str) -> None:
    """Print structured notes about what the spike learned."""
    print("\n" + "=" * 60)
    print("SPIKE FINDINGS")
    print("=" * 60)

    print(f"\nFinal URL: {page.url}")

    print("\nResponse text (first 500 chars):")
    print(response_text[:500])

    print(f"\nResponse length: {len(response_text)} characters")

    print("\nDOM structure hints:")
    print(f"  Total <div> elements: {page.locator('div').count()}")
    print(f"  Elements with role='textbox': {page.locator('[role="textbox"]').count()}")
    print(f"  Elements with data-testid: {page.locator('[data-testid]').count()}")
    print(f"  contenteditable elements: {page.locator('[contenteditable="true"]').count()}")

    print(f"\nBrowser will close in {INSPECTION_WAIT_SECONDS}s.")


def run_spike() -> None:
    """Main entry. Launches Chrome, exercises one round-trip, reports."""
    PROFILE_PATH.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_PATH),
            channel="chrome",
            headless=False,
            no_viewport=True,
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            print(f"Navigating to {CLAUDE_URL}")
            page.goto(CLAUDE_URL, wait_until="domcontentloaded")

            print(f"After goto, URL is: {page.url}")
            print(f"Page title: {page.title()}")

            if not is_logged_in(page):
                wait_for_login(page)

            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except PlaywrightTimeout:
                print(
                    "Network never idled (claude.ai may keep open connections). Continuing anyway."
                )

            print(f"Pre-input-search URL: {page.url}")
            print(f"Pre-input-search title: {page.title()}")

            try:
                input_selector = find_message_input(page)
            except RuntimeError as e:
                print(f"\nFAILED to find message input: {e}")
                print(f"\nCurrent URL: {page.url}")
                print(f"Current title: {page.title()}")
                body_text = page.locator("body").inner_text()[:500]
                print(f"\nBody text (first 500 chars):\n{body_text}")
                print(f"\nPausing {INSPECTION_WAIT_SECONDS}s. Inspect the browser now.")
                time.sleep(INSPECTION_WAIT_SECONDS)
                return

            send_message(page, input_selector, TEST_MESSAGE)

            print("Waiting for response...")
            response_text = wait_for_response(page)

            report_findings(page, response_text)

            time.sleep(INSPECTION_WAIT_SECONDS)
        finally:
            context.close()
            print("\nSpike complete.")


if __name__ == "__main__":
    run_spike()
