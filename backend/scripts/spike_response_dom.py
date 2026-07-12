"""Spike: dump the DOM structure of a claude.ai assistant response.

The Playwright transport scrapes div.font-claude-response with
text_content(), which concatenates every text node in the container,
UI chrome included, with no separators. A status chip glued onto the
first delimiter line broke the parser's line anchor. This spike
sends one planner-shaped message (tool instructions plus a delimited
output format, the shape that produced the chip) and prints the
response container's child tree with tags, classes, data-testids,
and text snippets, plus a text_content()/inner_text() comparison,
so the scrape fix picks its content selector from evidence.

Throwaway, like spike_claude_dom.py. Run from backend/ with:

    uv run python scripts/spike_response_dom.py
"""

from __future__ import annotations

import time

from app.core.config import get_settings
from patchright.sync_api import sync_playwright

CLAUDE_NEW_CHAT_URL = "https://claude.ai/new"
INPUT_SELECTOR = 'div[contenteditable="true"]'
RESPONSE_SELECTOR = "div.font-claude-response"
RESPONSE_WAIT_S = 90
STABLE_TICKS = 4

# Mimics the planner intro's shape: advertises a tool the web app
# does not natively have and demands delimited output. This is the
# combination that produced the "unavailable tool" chip.
TEST_MESSAGE = (
    "You are a planning assistant. You have one read-only tool: "
    "get_weak_topics. To call it, respond with exactly:\n"
    "---TOOL_CALL---\n"
    '{"name": "get_weak_topics", "args": {}}\n'
    "---END---\n"
    "Call the tool now. Do not add any other text."
)


def run_spike() -> None:
    """Open a chat, send the bait message, dump the response DOM."""
    settings = get_settings()

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(settings.chrome_profile_path),
            channel="chrome",
            headless=False,
            no_viewport=True,
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(CLAUDE_NEW_CHAT_URL, wait_until="domcontentloaded")
            page.wait_for_selector(INPUT_SELECTOR, timeout=30_000)

            box = page.locator(INPUT_SELECTOR).first
            box.click()
            box.fill(TEST_MESSAGE)
            page.keyboard.press("Enter")
            print("Message sent. Waiting for response to stabilize...")

            last_text = ""
            stable = 0
            deadline = time.time() + RESPONSE_WAIT_S
            while time.time() < deadline and stable < STABLE_TICKS:
                elements = page.locator(RESPONSE_SELECTOR).all()
                current = elements[-1].text_content() or "" if elements else ""
                if current and current == last_text:
                    stable += 1
                else:
                    stable = 0
                    last_text = current
                time.sleep(1)

            result = page.evaluate(
                """
                () => {
                    const els = document.querySelectorAll("div.font-claude-response");
                    if (!els.length) return {error: "no response container"};
                    const el = els[els.length - 1];
                    const lines = [];
                    const walk = (node, depth) => {
                        if (depth > 6) return;
                        for (const child of node.children) {
                            const cls = typeof child.className === "string"
                                ? child.className.slice(0, 90) : "";
                            const tid = child.getAttribute("data-testid") || "";
                            let own = "";
                            for (const n of child.childNodes) {
                                if (n.nodeType === 3) own += n.textContent;
                            }
                            own = own.trim().slice(0, 70);
                            lines.push(
                                "  ".repeat(depth)
                                + "<" + child.tagName.toLowerCase() + ">"
                                + (tid ? " testid='" + tid + "'" : "")
                                + (cls ? " cls='" + cls + "'" : "")
                                + (own ? " text='" + own + "'" : "")
                            );
                            walk(child, depth + 1);
                        }
                    };
                    walk(el, 0);
                    return {
                        tree: lines.join("\\n"),
                        textContent: (el.textContent || "").slice(0, 500),
                        innerText: (el.innerText || "").slice(0, 500),
                    };
                }
                """
            )

            print("\n" + "=" * 60)
            print("RESPONSE CONTAINER DOM TREE")
            print("=" * 60)
            print(result.get("tree", result))
            print("\n" + "=" * 60)
            print("text_content() (repr, first 500):")
            print(repr(result.get("textContent", "")))
            print("\ninner_text() (repr, first 500):")
            print(repr(result.get("innerText", "")))
        finally:
            context.close()
            print("\nSpike complete.")


if __name__ == "__main__":
    run_spike()
