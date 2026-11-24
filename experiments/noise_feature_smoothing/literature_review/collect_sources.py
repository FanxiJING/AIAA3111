"""Archive primary-source metadata without touching the frozen experiment."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.request import Request, urlopen
import time

HERE = Path(__file__).resolve().parent
SOURCES = [
    ("ranacher", "10.1080/13658816.2015.1086924",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC4786863/"),
    ("zeng", "10.1049/itr2.12243", None),
    ("dabiri", "10.1016/j.trc.2017.11.021", "https://arxiv.org/abs/1804.02386"),
    ("endo", "10.1007/978-3-319-31750-2_5", None),
    ("lu", "10.1016/j.trc.2025.105388",
     "https://www.sciencedirect.com/science/article/abs/pii/S0968090X25003924"),
    ("roberts", "10.1111/ecog.02881",
     "https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881"),
    ("farha", "10.1109/CVPR.2019.00369",
     "https://openaccess.thecvf.com/content_CVPR_2019/html/"
     "Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_"
     "Segmentation_CVPR_2019_paper.html"),
    ("skill", "10.48550/arXiv.2609.00065", "https://arxiv.org/abs/2609.00065"),
]
EXPECTED_TITLES = {
    "ranacher": "why gps makes distances bigger",
    "zeng": "addressing robust travel mode identification",
    "dabiri": "inferring transportation modes",
    "endo": "deep feature extraction from trajectories",
    "lu": "enhancing applicable travel mode identification",
    "roberts": "validation strategies for data",
    "farha": "ms-tcn: multi-stage temporal convolutional network",
}


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def fetch(url):
    request = Request(url, headers={"User-Agent": "GeoLife-course-evidence-audit/1.0"})
    with urlopen(request, timeout=25) as response:
        body = response.read()
        return body, response.geturl(), response.headers.get_content_type()


def main():
    target = HERE / "sources"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for key, doi, primary_url in SOURCES:
        record = {"key": key, "doi": doi, "requests": []}
        urls = [("crossref", "https://api.crossref.org/works/" + doi)]
        if primary_url:
            urls.append(("primary", primary_url))
        for kind, url in urls:
            item = {"kind": kind, "requested_url": url,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            try:
                body, resolved, content_type = fetch(url)
                suffix = ".json" if "json" in content_type else ".html"
                path = target / (key + "_" + kind + suffix)
                path.write_bytes(body)
                item.update({"ok": True, "resolved_url": resolved,
                             "path": str(path.relative_to(HERE)),
                             "sha256": hashlib.sha256(body).hexdigest(),
                             "bytes": len(body), "content_type": content_type})
                if kind == "crossref":
                    metadata = json.loads(body)["message"]
                    title = " ".join(metadata.get("title", [])).lower()
                    if key in EXPECTED_TITLES and EXPECTED_TITLES[key] not in title:
                        raise ValueError(f"DOI/title mismatch for {key}: {title}")
                    record["title_verified"] = True
                    record["metadata"] = {field: metadata[field] for field in
                        ("title", "author", "published", "published-online",
                         "published-print", "container-title", "abstract", "URL")
                        if field in metadata}
                else:
                    parser = PlainText()
                    parser.feed(body.decode("utf-8", errors="replace"))
                    (target / (key + "_primary.txt")).write_text(
                        " ".join(" ".join(parser.parts).split()), encoding="utf-8")
            except Exception as error:
                item.update({"ok": False, "error": f"{type(error).__name__}: {error}"})
            record["requests"].append(item)
        records.append(record)
        print(key, [(r["kind"], r["ok"]) for r in record["requests"]], flush=True)
        (HERE / "source_verification.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
