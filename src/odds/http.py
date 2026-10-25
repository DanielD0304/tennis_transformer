"""Kleiner JSON-Client für die Quoten-APIs: Wiederholungen, Rate-Limit und Zähler pro Host."""
import threading
import time
from collections import Counter
from urllib.parse import urlparse

import requests

from ..sources.common import USER_AGENT


class HttpError(RuntimeError):
    def __init__(self, url, status, text=""):
        super().__init__(f"HTTP {status} für {url}: {text[:200]}")
        self.status = status


class JsonClient:
    """
    get_json/post_json mit höchstens max_per_sec Anfragen pro Sekunde (thread-sicher).
    429 und 5xx werden mit Backoff wiederholt, andere Fehler sofort als HttpError gemeldet.
    """

    def __init__(self, max_per_sec=8.0, retries=6, timeout=30):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = USER_AGENT
        self.min_interval = 1.0 / max_per_sec if max_per_sec else 0.0
        self.retries = retries
        self.timeout = timeout
        self.calls = Counter()
        self._lock = threading.Lock()
        self._next = 0.0

    def _wait(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def request(self, method, url, **kwargs):
        last = None
        for attempt in range(self.retries):
            self._wait()
            with self._lock:
                self.calls[urlparse(url).netloc] += 1
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as e:
                last = e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r
            if r.status_code == 429 or r.status_code >= 500:
                last = HttpError(url, r.status_code, r.text)
                time.sleep(float(r.headers.get('Retry-After') or min(60, 2 ** (attempt + 1))))
                continue
            raise HttpError(url, r.status_code, r.text)
        raise last if isinstance(last, HttpError) else HttpError(url, 'Netzwerk', str(last))

    def get_json(self, url, params=None):
        return self.request('GET', url, params=params).json()

    def post_json(self, url, payload):
        return self.request('POST', url, json=payload).json()
