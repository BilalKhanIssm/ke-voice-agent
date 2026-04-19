# KE Voice Agent

Standalone LiveKit telephony worker with deterministic DTMF IVR language selection.

## Flow

1. Play bilingual IVR prompt.
2. Accept DTMF `1` (English) or `2` (Urdu).
3. Retry once for invalid/timeout input.
4. If still invalid/timeout, play polite goodbye and hang up.
5. Start agent session only after valid language selection.

## Required environment variables

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `LIVEKIT_AGENT_NAME`
- `DEEPGRAM_API_KEY`
- `OPENAI_API_KEY` when `LLM_PROVIDER=openai`
- `OPENROUTER_API_KEY` when `LLM_PROVIDER=openrouter`

Copy `.env.example` to `.env` and fill values.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.telephony.entrypoint start
```

## Playground test mode (no phone call cost)

Use this when you want to test the agent without dialing your LiveKit phone number.

1) Start worker in one terminal:

```bash
python -m app.telephony.entrypoint start
```

2) In another terminal, generate a dev room + participant token:

```bash
python scripts/livekit_dev_setup.py --dispatch-agent
```

The script prints:
- `WS_URL`
- `ROOM_NAME`
- `PARTICIPANT_TOKEN`

3) Open LiveKit Playground, connect with token:
- URL = `WS_URL`
- Token = `PARTICIPANT_TOKEN`

The agent is dispatched to the same room when `--dispatch-agent` is used.

## Docker run

```bash
docker compose up --build
```

## Test

```bash
pytest -q
```

## Observability markers

- `ivr.prompt`
- `ivr.input`
- `ivr.retry`
- `ivr.hangup`
- `lang.selected`
- `session.started`
