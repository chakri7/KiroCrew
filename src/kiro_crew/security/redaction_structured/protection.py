"""Protection masks: the Identifier VALUES a caller may hide from a later shape-based pass.

``protect(text)`` returns the redacted text, the resolved hits and the masks. A mask is
``(start, end, rule_id)`` over the input text. Masks are in text order, never overlap
each other, and never overlap a Secret-tier hit.

- The row's ``mask`` field decides how much of an Identifier hit is masked: ``value``
  the whole hit (a labelled value, a public parameter's value, a key ID, a
  placeholder); ``none`` nothing (a whole URL is a container, and an ARN is left to
  main's own ARN handling); ``sha_segment`` only the SHA segment of a commit
  permalink; ``image_payload`` a data URI's payload only when its media type is
  ``image/*``.
- A span that overlaps any Secret-tier hit a detector found is never masked. This is
  checked against the hits before resolve, so a Secret hit that resolve rule 1 dropped
  inside a structural Identifier still keeps its span unmasked.
- Parameter rows never list a credential parameter (``table.NEVER_MASK_PARAMS``), so
  ``client_secret``, ``code`` and the tokens are never masked.
"""

from __future__ import annotations

import bisect

from .charsets import cached
from .engine import render, resolve, scan
from .model import IDENTIFIER, SECRET, Hit
from .table import TABLE, Table

Mask = tuple[int, int, str]


def _modes(table: Table) -> dict[str, str]:
    return {row.id: row.mask for row in table.structures}


def _sha_segment(text: str, h: Hit) -> tuple[int, int] | None:
    """``scheme://host/owner/repo/(commit|blob|tree)/<sha>``: the ``<sha>`` segment."""
    q = text.find("://", h.start, h.end)
    if q == -1:
        return None
    pos = q + 3
    for _ in range(4):
        slash = text.find("/", pos, h.end)
        if slash == -1:
            return None
        pos = slash + 1
    end = text.find("/", pos, h.end)
    end = h.end if end == -1 else end
    return (pos, end) if end > pos else None


_IMAGE_TYPE = "image/"
#: Bitmap subtypes only. ``svg+xml`` (and any other image type) is a text document a
#: credential can sit in, so its payload is read by the later passes like any text.
_BITMAP_SUBTYPES = frozenset({"png", "jpeg", "jpg", "gif", "webp", "avif", "bmp"})
_DATA = "data:"
#: How far before a payload its ``data:`` header can start: a media type, up to eight
#: parameters and ``;base64,`` (the data URI row's own bounds).
_HEADER_REACH = 1024


def _image_payload(text: str, h: Hit) -> tuple[int, int] | None:
    """A data URI's payload when its media type is a bitmap image; ``None`` otherwise."""
    lo = max(0, h.start - _HEADER_REACH)
    window = text[lo : h.start].lower()
    head = window.rfind(_DATA)
    if head == -1:
        return None
    mime = head + len(_DATA)
    if window[mime : mime + len(_IMAGE_TYPE)] != _IMAGE_TYPE:
        return None
    sub = mime + len(_IMAGE_TYPE)
    end = sub
    while end < len(window) and window[end] not in ";,":
        end += 1
    if window[sub:end] not in _BITMAP_SUBTYPES:
        return None
    return h.start, h.end


#: Base64 characters decoded to read a bitmap's fixed header (24 bytes).
_HEAD_B64 = 32
#: JPEG markers that may follow ``FF D8 FF``: an APPn segment, a quantisation or
#: Huffman table, or a start of frame.
_JPEG_MARKERS = frozenset(range(0xE0, 0xF0)) | {0xDB, 0xC4, 0xC0, 0xC2}
#: BMP DIB header sizes (BITMAPCOREHEADER through BITMAPV5HEADER).
_BMP_DIB_SIZES = frozenset({12, 40, 52, 56, 64, 108, 124})


def well_formed_bitmap_head(text: str, start: int, end: int) -> bool:
    """Whether the base64 payload ``text[start:end]`` decodes to a bitmap's fixed header.

    A file signature alone is a few base64 characters anyone can type in front of a
    secret; the bytes after it are checked as that format lays them out: a PNG's
    ``IHDR`` chunk of 13 bytes, a JPEG's next marker, a GIF's version, a WebP's
    ``WEBP`` form type, a BMP's DIB header size.
    """
    import base64
    import binascii

    head = text[start : min(end, start + _HEAD_B64)]
    if len(head) < _HEAD_B64:
        return False
    try:
        b = base64.b64decode(head, validate=True)
    except (binascii.Error, ValueError):
        return False
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return b[8:16] == b"\x00\x00\x00\x0dIHDR"
    if b.startswith(b"\xff\xd8\xff"):
        return b[3] in _JPEG_MARKERS
    if b.startswith((b"GIF87a", b"GIF89a")):
        return True
    if b.startswith(b"RIFF"):
        return b[8:12] == b"WEBP"
    if b.startswith(b"BM"):
        return int.from_bytes(b[14:18], "little") in _BMP_DIB_SIZES
    return False


def _merged(spans: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    for s, e in sorted(spans):
        if ends and s < ends[-1]:
            ends[-1] = max(ends[-1], e)
        else:
            starts.append(s)
            ends.append(e)
    return starts, ends


def masks_for(text: str, hits: list[Hit], resolved: list[Hit], table: Table = TABLE) -> list[Mask]:
    """Masks for ``resolved`` Identifier hits; ``hits`` are the raw hits before resolve."""
    modes = cached(table, _modes)
    starts, ends = _merged([(h.start, h.end) for h in hits if h.tier == SECRET])

    def touches_secret(s: int, e: int) -> bool:
        i = bisect.bisect_right(starts, e - 1) - 1
        return i >= 0 and ends[i] > s

    out: list[Mask] = []
    last = -1
    for h in resolved:
        if h.tier != IDENTIFIER:
            continue
        mode = modes.get(h.rule_id, "value")
        if mode == "none":
            continue
        span: tuple[int, int] | None = (h.start, h.end)
        if mode == "sha_segment":
            span = _sha_segment(text, h)
        elif mode == "image_payload":
            span = _image_payload(text, h)
        if span is None or span[0] < last or touches_secret(*span):
            continue
        out.append((span[0], span[1], h.rule_id))
        last = span[1]
    return out


def protect(text: str, table: Table = TABLE) -> tuple[str, list[Hit], list[Mask]]:
    """``(redacted_text, resolved_hits, masks)`` for whole text."""
    hits = scan(text, final=True, table=table).hits
    if not hits:
        return text, [], []
    resolved = resolve(hits)
    return render(text, resolved, 0, len(text)), resolved, masks_for(text, hits, resolved, table)
