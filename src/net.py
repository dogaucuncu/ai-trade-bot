"""Shared TLS / networking helpers.

Centralises secure TLS setup. Certificate verification is routed through the
OS trust store via ``truststore`` so HTTPS keeps working on TLS-interception
networks (corporate proxy / AV SSL inspection) **without ever disabling
verification**. This replaces the old, unsafe ``ssl=False`` connectors that
left the live data feed open to man-in-the-middle price manipulation.

Usage
-----
>>> import aiohttp
>>> from src.net import verified_ssl_context
>>> connector = aiohttp.TCPConnector(ssl=verified_ssl_context())
"""

from __future__ import annotations

import ssl

from loguru import logger

_TRUSTSTORE_INSTALLED = False


def install_os_trust_store() -> None:
    """Idempotently route TLS verification through the OS certificate store.

    Safe to call from any entry point (main app, standalone scripts, tests).
    No-op if ``truststore`` is unavailable — verification then falls back to
    Python's default (certifi) bundle, which is fine off intercepted networks.
    """
    global _TRUSTSTORE_INSTALLED
    if _TRUSTSTORE_INSTALLED:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
        _TRUSTSTORE_INSTALLED = True
        logger.debug("[net] OS trust store injected for TLS verification")
    except Exception:
        # truststore optional — default verification still applies.
        pass


def verified_ssl_context() -> ssl.SSLContext:
    """Return a *verifying* SSL context (OS trust store when available).

    Always verifies certificates and hostnames. Use this for every outbound
    HTTPS client instead of ``ssl=False``.
    """
    install_os_trust_store()
    return ssl.create_default_context()
