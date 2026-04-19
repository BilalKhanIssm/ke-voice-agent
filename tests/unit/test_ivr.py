import pytest

from app.telephony.ivr import IVRConfig, select_language


class FakeIO:
    def __init__(self, keys):
        self.keys = list(keys)
        self.played: list[str] = []
        self.hung_up = False

    async def play(self, text: str) -> None:
        self.played.append(text)

    async def wait_for_dtmf(self, timeout_seconds: int):
        if self.keys:
            return self.keys.pop(0)
        return None

    async def hangup(self) -> None:
        self.hung_up = True


@pytest.mark.asyncio
async def test_ivr_selects_english():
    io = FakeIO(["1"])
    lang = await select_language(io, IVRConfig())
    assert lang == "en"
    assert not io.hung_up


@pytest.mark.asyncio
async def test_ivr_selects_urdu_on_retry():
    io = FakeIO(["9", "2"])
    lang = await select_language(io, IVRConfig(retries=1))
    assert lang == "ur"
    assert not io.hung_up


@pytest.mark.asyncio
async def test_ivr_hangup_after_retry_failure():
    io = FakeIO([None, None])
    lang = await select_language(io, IVRConfig(retries=1))
    assert lang is None
    assert io.hung_up
