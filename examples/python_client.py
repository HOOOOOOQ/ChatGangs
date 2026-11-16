#!/usr/bin/env python3
"""ChatGangs Python client example.

Run the API first (``python -m chatgangs``), then:

    python examples/python_client.py
    python examples/python_client.py --base-url http://localhost:8000 --api-key KEY

Requires ``requests`` (``pip install requests``); the standard library
``urllib`` fallback keeps the example runnable in a bare environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

try:  # optional nicety
    import requests  # type: ignore

    HAVE_REQUESTS = True
except Exception:  # pragma: no cover - example code
    HAVE_REQUESTS = False


class ChatGangsClient:
    """Minimal client for the ChatGangs REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None,
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------ http
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode() if payload is not None else None

        if HAVE_REQUESTS:
            response = requests.request(
                method, url, data=body, headers=self._headers(), timeout=self.timeout
            )
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code}: {response.text}")
            return response.json()

        request = urllib.request.Request(
            url, data=body, headers=self._headers(), method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                return json.loads(handle.read().decode())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise RuntimeError(f"{exc.code}: {exc.read().decode()}") from exc

    # --------------------------------------------------------------- methods
    def health(self) -> dict:
        return self._request("GET", "/api/health")

    def model_info(self) -> dict:
        return self._request("GET", "/api/model/info")

    def chat(self, message: str, session_id: str = "example", **kwargs) -> dict:
        payload = {"message": message, "session_id": session_id, **kwargs}
        return self._request("POST", "/api/chat", payload)

    def analyze(self, text: str, **kwargs) -> dict:
        return self._request("POST", "/api/nlp/analyze", {"text": text, **kwargs})

    def tokenize(self, text: str) -> dict:
        return self._request("POST", "/api/nlp/tokenize", {"text": text})

    def sentiment(self, text: str) -> dict:
        return self._request("POST", "/api/nlp/sentiment", {"text": text})

    def intent(self, text: str) -> dict:
        return self._request("POST", "/api/nlp/intent", {"text": text})

    def entities(self, text: str) -> dict:
        return self._request("POST", "/api/nlp/entities", {"text": text})

    def language(self, text: str) -> dict:
        return self._request("POST", "/api/nlp/language", {"text": text})

    def similarity(self, texts: list[str]) -> dict:
        return self._request("POST", "/api/nlp/similarity", {"texts": texts})

    def classify(self, text: str) -> dict:
        return self._request("POST", "/api/nlp/classify", {"text": text})

    def qa(self, question: str, top_k: int = 3) -> dict:
        return self._request("POST", "/api/nlp/qa", {"question": question, "top_k": top_k})

    def feedback(self, session_id: str, rating: int, comment: str | None = None) -> dict:
        return self._request(
            "POST",
            "/api/feedback",
            {"session_id": session_id, "rating": rating, "comment": comment},
        )

    def history(self, session_id: str, limit: int = 20) -> dict:
        return self._request("GET", f"/api/memory/{session_id}?limit={limit}")

    def delete_history(self, session_id: str) -> dict:
        return self._request("DELETE", f"/api/memory/{session_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ChatGangs example client")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    client = ChatGangsClient(args.base_url, args.api_key)

    print("== health ==")
    health = client.health()
    print(health["status"], health["version"], f"uptime={health['uptime_seconds']}s")

    print("\n== multi-turn conversation ==")
    first = client.chat("Halo, apa saja kemampuanmu?", session_id="example-1")
    print("bot :", first["reply"][:160], "...")
    print("intent:", first["intent"]["name"], first["intent"]["confidence"])

    second = client.chat("jelaskan lebih detail", session_id="example-1")
    print("follow-up:", second["context"]["is_followup"], "| turns:", second["context"]["turn_count"])
    print("bot :", second["reply"][:160], "...")

    print("\n== per-module analysis ==")
    text = "Pelayanan PT Maju Jaya bagus tapi pengirimannya lambat sekali."
    analysis = client.analyze(text)
    print("language :", analysis["language"]["code"], analysis["language"]["confidence"])
    print("intent   :", analysis["intent"]["name"], analysis["intent"]["confidence"])
    print("sentiment:", analysis["sentiment"]["label"], analysis["sentiment"]["score"])
    print("entities :", [(e["text"], e["label"]) for e in analysis["entities"]])
    print("tokens   :", [t["text"] for t in analysis["tokens"]][:10])
    print("keywords :", [k["term"] for k in analysis["keywords"]][:5])

    print("\n== knowledge base question answering ==")
    answer = client.qa("bagaimana cara memakai api")
    print("answered:", answer["answered"])
    print("answer  :", (answer["answer"] or "-")[:160], "...")

    print("\n== language detection across the 8 supported languages ==")
    for sample in [
        "Saya suka sekali produk ini",
        "I really like this product",
        "Saya sangat suka produk ini juga",
        "この製品がとても好きです",
        "이 제품을 정말 좋아합니다",
        "Me gusta mucho este producto",
        "Ich mag dieses Produkt sehr",
        "J'aime beaucoup ce produit",
    ]:
        result = client.language(sample)
        print(f"  {result['language']:>3} {result['confidence']:.2f}  {sample[:40]}")

    print("\n== feedback, history, deletion ==")
    client.feedback("example-1", rating=5, comment="great")
    history = client.history("example-1")
    print("messages stored:", len(history["messages"]))
    print("delete:", client.delete_history("example-1"))

    print("\n== model information ==")
    info = client.model_info()
    for task, artifact in info.get("artifacts", {}).items():
        print(f"  {task:>14}: trained={artifact.get('trained')} "
              f"accuracy={artifact.get('validation_accuracy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
