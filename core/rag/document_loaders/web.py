import asyncio
from typing import List, Sequence, Any, Dict, Literal, Iterator

import aiohttp
import requests
from pydantic import BaseModel, Field
from requests import Session

from core.logging import get_logger
from core.rag.document.document import MetadataMapping, Document
from core.rag.document_loaders.loader import DocumentLoader
from core.rag.document_loaders.utils import get_user_agent
from core.utils.utils import filter_kwargs_by_pydantic

logger = get_logger(__name__)

default_header_template = {
    "User-Agent": get_user_agent(),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}


class WebLoader(BaseModel, DocumentLoader):
    metadata: MetadataMapping = Field(default_factory=dict)
    excluded_embed_metadata_keys: List[str] = Field(default_factory=list)
    excluded_llm_metadata_keys: List[str] = Field(default_factory=list)

    web_urls: Sequence[str]
    continue_on_failure: bool = False
    auto_set_encoding: bool = True
    encoding: str | None = None
    requests_per_second: int = 2
    default_parser: Literal["html.parser", "lxml", "xml", "lxml-xml", "html5lib"] = "html.parser"
    requests_kwargs: Dict[str, Any] = Field(default_factory=dict)
    raise_for_status: bool = False
    bs_get_text_kwargs: Dict[str, Any] = Field(default_factory=dict)
    bs_kwargs: Dict[str, Any] = Field(default_factory=dict)
    session: Session

    def __init__(self,
                 web_urls: str | Sequence[str],
                 continue_on_failure: bool = False,
                 auto_set_encoding: bool = True,
                 encoding: str | None = None,
                 requests_per_second: int = 2,
                 default_parser: Literal["html.parser", "lxml", "xml", "lxml-xml", "html5lib"] = "html.parser",
                 requests_kwargs: Dict[str, Any] | None = None,
                 raise_for_status: bool = False,
                 bs_get_text_kwargs: Dict[str, Any] | None = None,
                 bs_kwargs: Dict[str, Any] | None = None,
                 session: Session | None = None,
                 header_template: Dict[str, str] | None = None,
                 verify_ssl: bool = True,
                 proxies: dict | None = None):
        if isinstance(web_urls, str):
            web_urls = [web_urls]
        if session is None:
            session = requests.Session()
            header_template = header_template or default_header_template.copy()
            if not header_template.get("User-Agent"):
                from fake_useragent import UserAgent  # type: ignore[import-untyped]
                header_template["User-Agent"] = UserAgent().random
            session.headers = dict(header_template)
            session.verify = verify_ssl
            if proxies:
                session.proxies.update(proxies)

        kwargs = filter_kwargs_by_pydantic(WebLoader, locals(), exclude_none=True)
        super().__init__(**kwargs)

    async def _fetch(self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5) -> str:
        async with aiohttp.ClientSession() as session:
            for i in range(retries):
                try:
                    async with session.get(url,
                                           headers=self.session.headers,
                                           ssl=None if self.session.verify else False,
                                           cookies=self.session.cookies.get_dict()) as response:
                        if self.raise_for_status:
                            response.raise_for_status()
                        return await response.text()
                except aiohttp.ClientConnectionError as e:
                    if i == retries - 1:
                        raise
                    else:
                        logger.warning(
                            f"Error fetching {url} with attempt {i + 1}/{retries}: {e}. Retrying.."
                        )
                        await asyncio.sleep(cooldown * backoff ** i)
        raise ValueError("retry count exceeded")

    async def _fetch_with_rate_limit(self, url: str, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            try:
                return await self._fetch(url)
            except Exception as e:
                if self.continue_on_failure:
                    logger.warning(
                        f"Error fetching {url}, skipping due to continue_on_failure=True"
                    )
                    return ""
                logger.exception(
                    f"Error fetching {url} and aborting, use continue_on_failure=True "
                    "to continue loading urls after encountering an error."
                )
                raise e

    async def fetch_all(self) -> Any:
        """Fetch all urls concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = []
        for url in self.web_urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            from tqdm.asyncio import tqdm_asyncio  # type: ignore[import-untyped]

            return await tqdm_asyncio.gather(
                *tasks, desc="Fetching pages", ascii=True, mininterval=1
            )
        except ImportError:
            logger.warning("For better logging of progress, `pip install tqdm`")
            return await asyncio.gather(*tasks)

    class Config:
        arbitrary_types_allowed = True

    def scrape_all(self) -> List[Any]:
        """Fetch all urls, then return soups for all results."""
        from bs4 import BeautifulSoup

        results = asyncio.run(self.fetch_all())
        final_results = []
        for i, result in enumerate(results):
            url = self.web_urls[i]
            parser = "xml" if url.endswith(".xml") else self.default_parser
            final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))

        return final_results

    def _scrape(self, url: str, bs_kwargs: dict | None = None) -> Any:
        from bs4 import BeautifulSoup

        parser = "xml" if url.endswith(".xml") else self.default_parser
        html_doc = self.session.get(url, **self.requests_kwargs)
        if self.raise_for_status:
            html_doc.raise_for_status()

        if self.encoding is not None:
            html_doc.encoding = self.encoding
        elif self.auto_set_encoding:
            html_doc.encoding = html_doc.apparent_encoding
        return BeautifulSoup(html_doc.text, parser, **(bs_kwargs or {}))

    def lazy_load(self) -> Iterator[Document]:
        for path in self.web_urls:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            yield Document(text=text, metadata=metadata)

    def load(self) -> List[Document]:
        results = self.scrape_all()
        docs = []
        for path, soup in zip(self.web_urls, results):
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            docs.append(Document(text=text, metadata=metadata))
        return docs


def _build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata
