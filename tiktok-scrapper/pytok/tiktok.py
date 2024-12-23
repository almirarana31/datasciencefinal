import logging
import os
import re
import asyncio
from typing import Optional

from browserforge.injectors.playwright import AsyncNewContext
from browserforge.headers import Browser as ForgeBrowser
from playwright.async_api import async_playwright
from undetected_playwright import Malenia

from .api.sound import Sound
from .api.user import User
from .api.search import Search
from .api.hashtag import Hashtag
from .api.video import Video
from .api.trending import Trending

from .exceptions import *
from .utils import LOGGER_NAME

os.environ["no_proxy"] = "127.0.0.1,localhost"

BASE_URL = "https://m.tiktok.com/"
DESKTOP_BASE_URL = "https://www.tiktok.com/"


class PyTok:
    _is_context_manager = False
    user = User
    search = Search
    sound = Sound
    hashtag = Hashtag
    video = Video
    trending = Trending
    logger = logging.getLogger(LOGGER_NAME)

    def __init__(
        self,
        logging_level: int = logging.WARNING,
        request_delay: Optional[int] = 0,
        headless: Optional[bool] = False,
        browser: Optional[str] = "chromium",
        manual_captcha_solves: Optional[bool] = False,
        log_captcha_solves: Optional[bool] = False,
    ):
        """Initialize PyTok for TikTok scraping."""
        self._headless = headless
        self._request_delay = request_delay
        self._browser = browser
        self._manual_captcha_solves = manual_captcha_solves
        self._log_captcha_solves = log_captcha_solves

        self.logger.setLevel(logging_level)

        # Add API classes
        User.parent = self
        Search.parent = self
        Sound.parent = self
        Hashtag.parent = self
        Video.parent = self
        Trending.parent = self

        self.request_cache = {}

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        fingerprint_options = {}
        if self._browser == "firefox":
            self._browser = await self._playwright.firefox.launch(headless=self._headless)
            fingerprint_options['browser'] = [ForgeBrowser("firefox")]
        elif self._browser == "chromium":
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
            fingerprint_options['browser'] = 'chrome'
        else:
            raise Exception("Browser not supported")

        self._context = await AsyncNewContext(self._browser, fingerprint_options=fingerprint_options)
        device_config = self._playwright.devices['Desktop Chrome']
        self._context = await self._browser.new_context(**device_config)
        # Removed viewport to avoid conflict
        self._context.set_default_timeout(60000)  # Set default timeout to 60 seconds
        await Malenia.apply_stealth(self._context)
        self._page = await self._context.new_page()

        await self._page.mouse.move(0, 0)

        self._requests = []
        self._responses = []

        self._page.on("request", lambda request: self._requests.append(request))
        self._page.on("response", lambda response: self._responses.append(response))

        self._user_agent = await self._page.evaluate("() => navigator.userAgent")
        self._is_context_manager = True

        # Handle CAPTCHA
        if self._manual_captcha_solves:
            await self.handle_captcha()

        return self


    async def handle_captcha(self):
        """Handle CAPTCHA manually."""
        try:
            captcha = await self._page.query_selector("div.secsdk-captcha-drag-icon")
            if captcha:
                print("CAPTCHA detected. Solve it manually.")
                await self._page.wait_for_timeout(60000)  # Wait 60 seconds for manual solving
        except Exception as e:
            self.logger.warning(f"CAPTCHA handling failed: {e}")

    async def request_delay(self):
        if self._request_delay is not None:
            await self._page.wait_for_timeout(self._request_delay * 1000)

    async def shutdown(self):
        """Shutdown PyTok properly."""
        try:
            await self._context.close()
            await self._browser.close()
            await self._playwright.stop()
        except Exception as e:
            self.logger.warning(f"Error during shutdown: {e}")

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.shutdown()
