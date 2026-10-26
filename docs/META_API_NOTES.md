# Meta Model API notes for Saans

Our own summary of what Saans uses. Full pages live in `.meta-docs/` (gitignored, refresh with the
commands at the bottom). Checked against the docs on 2026-10-26.

## Base URL, auth, models

- Base URL `https://api.meta.ai/v1`, OpenAI-compatible. Bearer key from `MODEL_API_KEY`.
- The OpenAI SDK does not read `MODEL_API_KEY` by itself: always pass `api_key=os.environ["MODEL_API_KEY"]`.
- `muse-spark-1.3`: reasoning + tool calling, takes text, image, video, audio, PDF. 1M-token context.
  We use it through the Responses API (`client.responses.create`).
- `sam-3.1`: segmentation and tracking for images and video, also over the Responses API.
- Rate limits are per team: 3,000 RPM for both models (Muse also 4M TPM). SAM is billed per image or
  per video frame, with a 50-frame minimum per video request.

## Files: upload once, reuse

- `client.files.create(purpose="user_data", file=f)` returns an id starting with `file-`.
- Upload cap 1 GiB per file; 100 GiB of storage per team. Inline (base64) media is capped at 50 MB.
- Uploads never expire unless you pass `expires_after={"anchor": "created_at", "seconds": 3600..2592000}`.
  We delete the file ourselves (`client.files.delete(id)`) when a report is done.
- MP4 uploads without faststart get remuxed server-side (no re-encode, frames unchanged).

## Muse with tools (our agent loop, `backend/analyzer.py`)

```python
client.responses.create(
    model="muse-spark-1.3",
    instructions=SYSTEM_PROMPT,             # not carried by previous_response_id: resend every turn
    input=[{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "..."},
        {"type": "input_file", "file_id": file_id},   # input_video / input_image with file_id also work
    ]}],
    tools=[{"type": "function", "name": "segment", "description": "...",
            "parameters": {...json schema...}, "strict": True}],
    reasoning={"summary": "auto"},          # optional "effort": minimal..max ("none" is a 400)
    previous_response_id=prev_id,           # omit on the first turn; tools/reasoning must be resent too
)
```

- Tool definitions are flat (`type`, `name`, `description`, `parameters`, `strict`), not nested
  under `function`.
- `tool_choice` accepts only `"auto"`. Anything else is a 400, so you cannot force a tool call.
  Handle a plain-text reply too.
- `parallel_tool_calls` defaults to true: one response can hold several `function_call` items.
- A function call item has `name`, `arguments` (JSON string, not guaranteed valid: parse and
  validate it), and `call_id`. Reply with `{"type": "function_call_output", "call_id": ..., "output": "<string>"}`.
- With `previous_response_id` you send only the new items. Without it you resend the whole history,
  and every output must match a call in that same request.
- `strict: true` makes the server check the schema: every object needs `additionalProperties: false`
  and must list all its properties in `required`. No `oneOf`/`allOf`, no `anyOf` at the root.
- Reasoning summaries come back as `reasoning` output items with a `summary` array. It is often
  empty. Raw chain of thought is never returned.
- Structured output on Responses uses `text.format` (not `response_format`). We don't use it: the
  final answer comes through the `submit_findings` tool instead.
- `max_output_tokens` covers reasoning plus visible output.

## SAM: requests

One user message: an `input_text` with one short noun phrase (one concept, no lists or "and"), plus one media part.

| Media | Part we send |
|---|---|
| Image | `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}` (or a public URL) |
| Video | `{"type": "input_video", "file_id": "file-..."}` (or `video_url`) |

- `stream=True` (recommended for video; one delta per frame) and
  `metadata={"include_confidence": "true"}`. Metadata values are strings, not booleans.
- Optional `metadata.mask_encoding`: `"lossless"` (default) or `"one_bit"` (smaller).
- Video: MP4 or MOV, H.264, H.265 or MPEG-4. No VP9 or WebM. Up to 15,000 frames (~10 min).
- At most 16 objects per prompt. Several object types means several requests.
- Box and point prompts exist (`build_object_prompt` in the parser), with `propagation_direction`
  for video. We don't use them yet.
- A bad prompt or bad metadata returns 400 (don't retry unchanged); an unknown model returns 404 `model_not_found`.

## SAM: output and parsing

- One line per frame in a single `output_text` lane: `<Nf>` then records separated by commas.
  A record is `id<|box;x1;y1;x2;y2;w;h[;c]|><|mask;x=0;y=0[;c];data=H,W,<enc>payload|>`.
- Box corners are in source pixels (x2/y2 inclusive on the wire; the parser makes them half-open).
  `w`/`h` give the frame size after rotation.
- The mask is a small raster covering the box (example in the docs: a 60×27 box has a `data=27,60`
  mask). H comes before W. Payload is base85 with `~` (lossless) or `!` (one-bit).
- Object ids stay stable across frames while the object is in view. They are sparse, so key by id.
- Frames with no objects have no line. Concept prompts return frames in ascending order.
- Zero matches is a normal result: the stream completes with empty output and no delta events.
- Never parse by hand. In Python:

```python
from meta_sam_parser import (parse_responses_stream, video_segmentation_format,
                             image_segmentation_format, decode_mask_to_raster)
result = await parse_responses_stream(async_events, video_segmentation_format()).final_result()
for r in result.records:   # SegmentationBoxRecord / SegmentationMaskRecord / SegmentationTextRecord
    r.object_id, r.frame.frame_index, r.confidence          # confidence is None when not requested
```

  The parser only takes an async iterable. `backend/sam.py` wraps the sync SDK stream in an async
  generator and runs it with `asyncio.run`. Check `result.diagnostics` for skipped bad lines.
- `decode_mask_to_raster(mask_record.mask)` gives H*W bytes of 0/1 (row-major).
  `decode_mask_to_rle` and `decode_mask_to_svg_path` also exist.
- Frontend: `@meta-sam/parser` has the same API (`parseImageStream`, `decodeMaskToRaster`).

## Errors worth handling

- 429: back off and honor `Retry-After`. 500: retry with backoff. 503 (`service_overloaded`,
  `backend_unavailable`, `server_shutting_down`): always retryable, honor `Retry-After`.
- A mid-stream shutdown arrives as an `error` event with `code: server_shutting_down`; retry the whole request.
- 413 on a non-streaming request with `store=true` and large inline base64 media (~16 MB
  persisted): upload the file instead or set `store=false`.

## Unverified assumptions (confirm once the key works)

1. **Unverified.** SAM images go inline as base64 `image_url`. The SAM page lists only `image_url`
   for `input_image` (`file_id` is documented for `input_video` only), so our upload is reused for Muse only.
2. **Unverified.** One uploaded `file_id` works for both `muse-spark-1.3` and `sam-3.1`. The docs
   say to reuse ids across requests, but no example mixes models.
3. **Unverified.** Muse gets the media as `input_file` + `file_id`. The video page recommends it;
   `input_video` + `file_id` is the documented alternative.
4. **Unverified.** Zero matches: the docs say the stream ends with empty output and no deltas.
   It's unclear whether an `output_text.done` lane event still arrives. `sam.segment` accepts both.
5. **Partly confirmed by the docs example.** Masks are box-local rasters (box size, origin at
   x1,y1). We stretch them over the box. Check that outlines line up on a real clip.
6. **Unverified.** Reasoning summary elements have a `.text` field (OpenAI SDK `Summary` type). The
   docs only say "a summary array", which may be empty.
7. **Confirmed limitation, behavior unverified.** `tool_choice` can't force `submit_findings`. We send a
   "submit now" message on the last turn and nudge after a text-only reply. Check that Muse complies.
8. **Unverified.** Strict tool schemas: the docs don't say whether `minimum`/`maximum` are allowed,
   so confidence (0 to 1) is checked in code, not in the schema.
9. **Unverified.** Default reasoning effort is fast enough on a 20 s clip within `MUSE_TIMEOUT_S=120`.
   If not, lower `reasoning.effort` or stream.
10. **Unverified.** The cookbook agent-loop pages have no code. Our loop (chain with
    `previous_response_id`, resend `instructions`/`tools`/`reasoning`) follows the tool-calling and
    Responses pages.

## Refresh the cache

`https://dev.meta.ai/llms.txt` indexes every page; each page is also served as `<url>.md`.

```bash
curl -sfL https://dev.meta.ai/llms.txt -o .meta-docs/llms.txt
curl -sfL https://dev.meta.ai/docs/sam/segmenting.md -o .meta-docs/sam__segmenting.md   # etc.
```
