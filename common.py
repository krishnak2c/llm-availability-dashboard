import urllib.request

# Auth headers stripped on cross-host redirects (compared case-insensitively).
_STRIPPED_HEADERS = frozenset({"authorization", "x-goog-api-key", "x-api-key"})


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and req.host != new_req.host:
            for hdrs in (new_req.unredirected_hdrs, new_req.headers):
                for k in [k for k in hdrs if k.lower() in _STRIPPED_HEADERS]:
                    del hdrs[k]
        return new_req


_opener = urllib.request.build_opener(SafeRedirectHandler())


def _is_free(p, key):
    val = p.get(key)
    if val is None:
        return False
    try:
        return float(val) == 0
    except (ValueError, TypeError):
        return False
