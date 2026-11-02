"""Receives RTP/WebSocket/SIPREC streams or recording files, decodes, splits channels,
resamples and publishes audio.frames (docs/01 §2). Streaming transport arrives in P4.
"""

SERVICE_NAME = "ingest_gateway"
