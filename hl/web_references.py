"""Shared fetch requirements for external web references."""

from __future__ import annotations

from urllib.parse import urlparse


DEFAULT_WEB_USER_AGENT = "HarnessEvolver/1.0"
WECHAT_ARTICLE_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; 23013RK75C Build/TKQ1.220829.002; wv) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/116.0.0.0 "
    "Mobile Safari/537.36 XWEB/1160043 MMWEBSDK/20240501 "
    "MicroMessenger/8.0.49.2600(0x28003133) Process/tools WeChat/arm64 Weixin "
    "NetType/WIFI Language/zh_CN ABI/arm64"
)


def is_wechat_article_url(url: str) -> bool:
    return (urlparse(url).hostname or "") == "mp.weixin.qq.com"


def user_agent_for_url(url: str) -> str:
    if is_wechat_article_url(url):
        return WECHAT_ARTICLE_USER_AGENT
    return DEFAULT_WEB_USER_AGENT


def request_headers_for_url(url: str) -> dict[str, str]:
    if is_wechat_article_url(url):
        return {
            "User-Agent": WECHAT_ARTICLE_USER_AGENT,
            "Referer": "https://mp.weixin.qq.com/",
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "*/*;q=0.8"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
    return {"User-Agent": DEFAULT_WEB_USER_AGENT}
