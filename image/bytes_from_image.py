from io import BytesIO


def extract_image_bytes_from_content_response(response) -> list:
    """Pull raw JPEG/PNG bytes from a Gemini generate_content image response."""
    collected = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                collected.append(inline.data)
                continue
            if hasattr(part, "as_image"):
                try:
                    pil_img = part.as_image()
                    buf = BytesIO()
                    pil_img.save(buf, format="JPEG", quality=92)
                    collected.append(buf.getvalue())
                except Exception:
                    pass
    return collected