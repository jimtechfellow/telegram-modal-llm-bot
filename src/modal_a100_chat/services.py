"""Composition only: Telegram does not implement research or provider selection."""

from search_bot.pi_web_access import PiWebAccess
from search_bot.service import SearchBot

from .backend import ModalBackend
from .runner import Runner


def create_runner(config, store):
    search = PiWebAccess(
        command=config.search_mcp_command,
        timeout_seconds=config.search_timeout_seconds,
        num_results=config.search_max_results,
    )
    return Runner(config, store, ModalBackend(config), search_bot=SearchBot(search))
