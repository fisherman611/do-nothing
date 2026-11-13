# Vibi API Reference

> Vibi API: text to speech, multi-speaker dialogue, voice changer, dubbing, speech to text and voice management over plain HTTPS + JSON. Generation endpoints are asynchronous: the create call returns 202 with a task id; poll the matching history endpoint until status is completed, then download the result URL.

Interactive version: https://vibi.pro/docs

## Getting started

- Base URL: `https://api.vibi.pro`
- Authentication: send your API key in the `xi-api-key` header on every request. It is shown on https://vibi.pro/docs while you are signed in. The few account endpoints marked BearerAuth take the web session token in `Authorization: Bearer <token>` instead.
- Asynchronous flow: `POST` a generation request, get `202 {"id": "...", "status": "pending"}`, then `GET /v1/history/{id}` (dubbing, voice changer and speech to text have their own history endpoints) until `status` is `completed`, then fetch `result.audio_url`. Statuses: pending, processing, completed, failed.
- Models: ElevenLabs `eleven_v3` (latest, best quality; the default when `model_id` is omitted), MiniMax `speech-2.8-hd` (best) or `speech-2.8-turbo` (faster), CapCut `capcut`. `GET /v1/models` lists them.
- Credits are deducted per character of input text; `GET /v1/auth/me` returns `credit_balance`.

## API Keys

### Get API key

`GET /v1/api-keys`

Docs: https://vibi.pro/docs/api-keys/get-api-key

Get current user's API key

**Authentication:** `Authorization: Bearer YOUR_TOKEN` header (web session token)

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/api-keys" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/api-keys', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer YOUR_TOKEN'
  }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/api-keys',
    headers={'Authorization': 'Bearer YOUR_TOKEN'},
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "api_key": "sk_abc..."
}
```

### Regenerate API key

`POST /v1/api-keys/regenerate`

Docs: https://vibi.pro/docs/api-keys/regenerate-api-key

Regenerate API key

**Authentication:** `Authorization: Bearer YOUR_TOKEN` header (web session token)

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/api-keys/regenerate" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/api-keys/regenerate', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_TOKEN'
  }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/api-keys/regenerate',
    headers={'Authorization': 'Bearer YOUR_TOKEN'},
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "api_key": "sk_abc...",
  "message": "API key regenerated successfully"
}
```

## Auth

### Get user, credits...

`GET /v1/auth/me`

Docs: https://vibi.pro/docs/auth/get-me

Returns the authenticated user's profile including credit balance.
This is the primary endpoint for retrieving the user's remaining credits.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/auth/me" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/auth/me', {
  method: 'GET',
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/auth/me',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "name": "John Doe",
  "avatar_url": "https://lh3.googleusercontent.com/...",
  "credit_balance": 4000,
  "role": "user"
}
```

## Text to speech

### Text to speech

`POST /v1/text-to-speech/{voice_id}`

Docs: https://vibi.pro/docs/tts/text-to-speech

Convert text to speech with a specific voice. Supports ElevenLabs, MiniMax, and CapCut providers.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `voice_id` | string | yes | Voice ID. For ElevenLabs: use the voice ID from the voice library. For MiniMax: use numeric voice_id (e.g. '226905123659939'), uniq_id (e.g. 'English_ManWithDeepVoice'), or UUID for cloned voices. For CapCut: use the voice_id from the CapCut voice library (see "List system voices" under CapCut Voices). |

**Request body** (`application/json`, required)

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | The text to convert to speech |
| `model_id` | string | no | ElevenLabs: eleven_v3 (latest, best quality; used when omitted), eleven_multilingual_v2, eleven_flash_v2_5 and other ElevenLabs model ids. MiniMax: speech-2.8-hd (best quality) or speech-2.8-turbo (faster). CapCut: 'capcut' (the default when provider is 'capcut'). GET /v1/models?provider=... lists the valid ids. Default: eleven_v3. |
| `provider` | string | no | TTS provider to use: 'elevenlabs' (default), 'minimax', or 'capcut' Default: elevenlabs. |
| `language_code` | string | yes | Language code from elevenlabs or language name from minimax |
| `voice_settings` | object | no | Voice settings. Available fields depend on the provider and model. |
| `voice_settings.stability` | number | no | [ElevenLabs] Stability (0.0 - 1.0). eleven_v3 accepts only 0.0 (creative), 0.5 (natural) or 1.0 (robust). |
| `voice_settings.similarity_boost` | number | no | [ElevenLabs] Similarity boost (0.0 - 1.0). Not used by eleven_v3. |
| `voice_settings.style` | number | no | [ElevenLabs] Style (0.0 - 1.0). Only supported by some models (e.g. eleven_multilingual_v2), not eleven_v3. |
| `voice_settings.use_speaker_boost` | boolean | no | [ElevenLabs] Enable speaker boost. Only supported by some models (e.g. eleven_multilingual_v2), not eleven_v3. |
| `voice_settings.speed` | number | no | Speed. ElevenLabs: 0.7-1.2 (default 1.0). MiniMax & CapCut: 0.5-2.0 (default 1.0). |
| `voice_settings.pitch` | integer | no | [MiniMax & CapCut] Pitch adjustment (-12 to 12, default 0) |
| `voice_settings.vol` | number | no | [MiniMax] Volume (0.01 - 10, default 1.0) |
| `export_transcript` | boolean | no | Generate SRT transcript alongside the audio. Adds ~15% to credit cost. Default: false. |

**Example (curl)**

```bash
# ElevenLabs (default provider)
curl -X POST "https://api.vibi.pro/v1/text-to-speech/{voice_id}" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "model_id": "eleven_v3",
    "language_code": "en",
    "voice_settings": {
      "stability": 0.5,
      "speed": 1.0
    }
  }'

# MiniMax provider
curl -X POST "https://api.vibi.pro/v1/text-to-speech/{voice_id}" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "model_id": "speech-2.8-hd",
    "provider": "minimax",
    "language_code": "English",
    "voice_settings": {
      "speed": 1.0,
      "pitch": 0,
      "vol": 1.0
    }
  }'

# CapCut provider
curl -X POST "https://api.vibi.pro/v1/text-to-speech/{voice_id}" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "provider": "capcut",
    "language_code": "en",
    "voice_settings": {
      "speed": 1.0,
      "pitch": 0
    }
  }'
```

**Example (JavaScript)**

```javascript
// ElevenLabs (default provider)
const response = await fetch('https://api.vibi.pro/v1/text-to-speech/{voice_id}', {
  method: 'POST',
  headers: {
    'xi-api-key': 'YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: 'Hello world',
    model_id: 'eleven_v3',
    provider: 'elevenlabs',
    language_code: 'en',
    voice_settings: {
      stability: 0.5,
      speed: 1.0
    }
  })
});
const data = await response.json();

// MiniMax provider
const response = await fetch('https://api.vibi.pro/v1/text-to-speech/{voice_id}', {
  method: 'POST',
  headers: {
    'xi-api-key': 'YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: 'Hello world',
    model_id: 'speech-2.8-hd',
    provider: 'minimax',
    language_code: 'English',
    voice_settings: { speed: 1.0, pitch: 0, vol: 1.0 }
  })
});
const data = await response.json();

// CapCut provider
const response = await fetch('https://api.vibi.pro/v1/text-to-speech/{voice_id}', {
  method: 'POST',
  headers: {
    'xi-api-key': 'YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: 'Hello world',
    provider: 'capcut',
    language_code: 'en',
    voice_settings: { speed: 1.0, pitch: 0 }
  })
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

# ElevenLabs (default provider)
response = requests.post(
    'https://api.vibi.pro/v1/text-to-speech/{voice_id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    json={
        'text': 'Hello world',
        'model_id': 'eleven_v3',
        'provider': 'elevenlabs',
        'language_code': 'en',
        'voice_settings': {
            'stability': 0.5,
            'speed': 1.0
        }
    }
)
data = response.json()

# MiniMax provider
response = requests.post(
    'https://api.vibi.pro/v1/text-to-speech/{voice_id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    json={
        'text': 'Hello world',
        'model_id': 'speech-2.8-hd',
        'provider': 'minimax',
        'language_code': 'English',
        'voice_settings': {
            'speed': 1.0,
            'pitch': 0,
            'vol': 1.0
        }
    }
)
data = response.json()

# CapCut provider
response = requests.post(
    'https://api.vibi.pro/v1/text-to-speech/{voice_id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    json={
        'text': 'Hello world',
        'provider': 'capcut',
        'language_code': 'en',
        'voice_settings': {
            'speed': 1.0,
            'pitch': 0
        }
    }
)
data = response.json()
```

**Responses**

- `202` Accepted - Task queued for processing

```json
{
  "id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "status": "pending"
}
```

### Get history detail

`GET /v1/history/{id}`

Docs: https://vibi.pro/docs/tts/get-history-detail

Get details of a specific text to speech task, dialogue task by ID, including status and audio URL

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID |

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/history/{id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/history/{id}', {
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const task = await response.json();
console.log(task.audio_url);
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/history/{id}',
    headers={'xi-api-key': 'YOUR_API_KEY'}
)
task = response.json()
```

**Responses**

- `200` Status can be pending, processing, completed, or failed

```json
{
  "id": "id_of_the_task",
  "user_id": "id_of_the_user",
  "status": "processing",
  "progress": 60,
  "provider": "elevenlabs",
  "text": "Hello world!",
  "voice_id": "UgBBYS2sOqTuMpoF3BR0",
  "model_id": "eleven_v3",
  "metadata": {
    "voice_settings": {
      "stability": 0.5,
      "speed": 1
    }
  },
  "result": {
    "audio_url": "https://api.vibi.pro/audio/id_of_the_task.mp3"
  },
  "error": string | null,
  "characters_used": 12,
  "credits_deducted": 12,
  "created_at": "2026-01-24T07:42:13.641779Z",
  "updated_at": "2026-01-24T07:42:16.432458Z"
}
```

### Get history list

`GET /v1/history`

Docs: https://vibi.pro/docs/tts/get-history

Get list of text to speech tasks, dialogue tasks across all providers

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page_size` | int | no | Number of tasks per page (max 100) Example: 30. |
| `page` | int | no | Page number (0-indexed) Example: 0. |

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/history?page_size=30&page=0" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/history?page_size=30&page=0', {
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const data = await response.json();
console.log(data.tasks);
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/history',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    params={'page_size': 30, 'page': 0}
)
history = response.json()
```

**Responses**

- `200` Success. The provider field indicates which TTS engine was used: 'elevenlabs', 'minimax', or 'capcut'.

```json
{
  "tasks": [
    {
      "id": "id_of_the_task",
      "user_id": "id_of_the_user",
      "status": "completed",
      "progress": 100,
      "provider": "elevenlabs",
      "text": "Hello world!",
      "voice_id": "UgBBYS2sOqTuMpoF3BR0",
      "model_id": "eleven_v3",
      "metadata": {
        "language_code": "en", // Elevenlabs use language code
        "voice_settings": {
          "stability": 0.5,
          "speed": 1
        }
      },
      "result": {
        "audio_url": "https://api.vibi.pro/audio/id_of_the_task.mp3"
      },
      "error": string | null,
      "characters_used": 12,
      "credits_deducted": 12,
      "created_at": "2026-01-24T07:42:13.641779Z",
      "updated_at": "2026-01-24T07:42:16.432458Z"
    },
    {
      "id": "id_of_another_task",
      "user_id": "id_of_the_user",
      "status": "completed",
      "progress": 100,
      "provider": "minimax",
      "text": "Hello from MiniMax!",
      "voice_id": "voice-id-uuid",
      "model_id": "speech-2.8-hd",
      "metadata": {
        "voice_settings": {
          "speed": 1.0,
          "pitch": 0,
          "vol": 1.0
        },
        "language_code": "English" // Minimax use full language name
      },
      "result": {
        "audio_url": "https://api.vibi.pro/audio/id_of_another_task.mp3"
      },
      "error": null,
      "characters_used": 19,
      "credits_deducted": 12,
      "created_at": "2026-01-24T08:00:00.000000Z",
      "updated_at": "2026-01-24T08:00:05.000000Z"
    }
  ],
  "has_more": false
}
```

### Delete history

`DELETE /v1/history/{id}`

Docs: https://vibi.pro/docs/tts/delete-history

Delete a text to speech task from history

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | History item ID |

**Example (curl)**

```bash
curl -X DELETE "https://api.vibi.pro/v1/history/{id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/history/{id}', {
  method: 'DELETE',
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.delete(
    'https://api.vibi.pro/v1/history/{id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "message": "deleted"
}
```

### Retry task

`POST /v1/history/{id}/retry`

Docs: https://vibi.pro/docs/tts/retry-task

Retry a failed or completed TTS task. Re-uses the original text and settings, re-deducts credits. Supports regular TTS, SRT, and dialogue tasks.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID to retry |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/history/{id}/retry" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/history/{id}/retry', {
  method: 'POST',
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/history/{id}/retry',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `202` Accepted - Task re-queued for processing

```json
{
  "id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "status": "pending"
}
```

## Dialogue

### Create dialogue

`POST /v1/dialogue`

Docs: https://vibi.pro/docs/dialogue/create-dialogue

Convert a multi-speaker dialogue script to speech. Each speaker can use a different provider, voice, model, and settings.
The dialogue text format is "SpeakerName: text..." - each line starting with a speaker name followed by a colon begins a new turn.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Request body** (`application/json`, required)

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Dialogue script in "SpeakerName: text" format. Each line starting with "Name:" begins a new turn. Multi-line text per turn is supported (continuation lines without "Name:" prefix). At least 2 different speakers must appear in the text. Example: Alice: Hello, how are you? Bob: I'm doing great, thanks for asking! Alice: That's wonderful to hear. |
| `speakers` | object | yes | Map of speaker name -> TTS configuration. Keys must match the speaker names used in the text. At least 2 speakers are required. Each speaker is fully independent with their own provider, voice, model, and settings. |
| `speakers.[speaker_name]` | object | no | TTS configuration for this speaker |
| `pause_between_turns` | number | no | Silence duration between turns in seconds (0-3) Default: 0. |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/dialogue" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice: Hello, how are you today?\nBob: I am doing great, thanks for asking!\nAlice: That is wonderful to hear.\nCarol: Hi both, nice to meet you.",
    "speakers": {
      "Alice": {
        "provider": "elevenlabs",
        "voice_id": "rv30Fd6w5bnbL0kHzWlr",
        "model_id": "eleven_v3",
        "language_code": "en",
        "voice_settings": {
          "stability": 0.5,
          "speed": 1.0
        }
      },
      "Bob": {
        "provider": "minimax",
        "voice_id": "English_ManWithDeepVoice",
        "model_id": "speech-2.8-hd",
        "language_code": "English",
        "voice_settings": {
          "speed": 1.0,
          "pitch": 0,
          "vol": 1.0
        }
      },
      "Carol": {
        "provider": "capcut",
        "voice_id": "7369525170723099153",
        "model_id": "capcut",
        "language_code": "en",
        "voice_settings": { "speed": 1.0, "pitch": 0 }
      }
    },
    "pause_between_turns": 0.5
  }'
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/dialogue', {
  method: 'POST',
  headers: {
    'xi-api-key': 'YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: 'Alice: Hello, how are you today?\nBob: I am doing great, thanks for asking!\nAlice: That is wonderful to hear.\nCarol: Hi both, sorry I am late!',
    speakers: {
      Alice: {
        provider: 'elevenlabs',
        voice_id: 'rv30Fd6w5bnbL0kHzWlr',
        model_id: 'eleven_v3',
        language_code: 'en',
        voice_settings: { stability: 0.5, similarity_boost: 0.75, speed: 1.0 }
      },
      Bob: {
        provider: 'minimax',
        voice_id: 'English_ManWithDeepVoice',
        model_id: 'speech-2.8-hd',
        language_code: 'English',
        voice_settings: { speed: 1.0, pitch: 0, vol: 1.0 }
      },
      Carol: {
        provider: 'capcut',
        voice_id: '7369525170723099153',
        model_id: 'capcut',
        language_code: 'en',
        voice_settings: { speed: 1.0, pitch: 0 }
      }
    },
    pause_between_turns: 0.5
  })
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/dialogue',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    json={
        'text': 'Alice: Hello, how are you today?\nBob: I am doing great, thanks for asking!\nAlice: That is wonderful to hear.\nCarol: Hi both, sorry I am late!',
        'speakers': {
            'Alice': {
                'provider': 'elevenlabs',
                'voice_id': 'rv30Fd6w5bnbL0kHzWlr',
                'model_id': 'eleven_v3',
                'language_code': 'en',
                'voice_settings': {
                    'stability': 0.5,
                    'speed': 1.0
                }
            },
            'Bob': {
                'provider': 'minimax',
                'voice_id': 'English_ManWithDeepVoice',
                'model_id': 'speech-2.8-hd',
                'language_code': 'English',
                'voice_settings': {
                    'speed': 1.0,
                    'pitch': 0,
                    'vol': 1.0
                }
            },
            'Carol': {
                'provider': 'capcut',
                'voice_id': '7369525170723099153',
                'model_id': 'capcut',
                'language_code': 'en',
                'voice_settings': {'speed': 1.0, 'pitch': 0}
            }
        },
        'pause_between_turns': 0.5
    }
)
data = response.json()
```

**Responses**

- `202` Accepted - Dialogue task queued for processing

```json
{
  "id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "status": "pending"
}
```

## Voice Changer

### Voice changer

`POST /v1/voice-changer`

Docs: https://vibi.pro/docs/voice-changer/voice-changer

Convert speech audio to a different voice (speech-to-speech). Upload an audio file and specify a target voice.
Only ElevenLabs voices are supported. Processing is async - poll the history endpoint for results.
Credits are calculated based on audio duration: ~1000 credits per minute.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Request body** (`multipart/form-data`, required)

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | file | yes | Source audio file to convert (max 50MB, max 5 minutes) |
| `voice_id` | string | yes | Target ElevenLabs voice ID |
| `model_id` | string | no | ElevenLabs STS model ID Default: eleven_multilingual_sts_v2. |
| `remove_background_noise` | string | no | Set to 'true' to apply background noise removal Default: false. |
| `voice_settings` | string (JSON) | no | Voice settings as a JSON string. Available fields: - stability (number): 0.0 - 1.0 - similarity_boost (number): 0.0 - 1.0 - style (number): 0.0 - 1.0 - use_speaker_boost (boolean) |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/voice-changer" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "audio=@recording.mp3" \
  -F "voice_id=rv30Fd6w5bnbL0kHzWlr" \
  -F "model_id=eleven_multilingual_sts_v2" \
  -F "remove_background_noise=true" \
  -F 'voice_settings={"stability":0.5,"similarity_boost":0.75}'
```

**Example (JavaScript)**

```javascript
const formData = new FormData();
formData.append('audio', audioFile);
formData.append('voice_id', 'rv30Fd6w5bnbL0kHzWlr');
formData.append('model_id', 'eleven_multilingual_sts_v2');
formData.append('remove_background_noise', 'true');
formData.append('voice_settings', JSON.stringify({
  stability: 0.5,
  similarity_boost: 0.75
}));

const response = await fetch('https://api.vibi.pro/v1/voice-changer', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' },
  body: formData
});
const data = await response.json();
```

**Example (Python)**

```python
import json
import requests

with open('recording.mp3', 'rb') as f:
    response = requests.post(
        'https://api.vibi.pro/v1/voice-changer',
        headers={'xi-api-key': 'YOUR_API_KEY'},
        files={'audio': f},
        data={
            'voice_id': 'rv30Fd6w5bnbL0kHzWlr',
            'model_id': 'eleven_multilingual_sts_v2',
            'remove_background_noise': 'true',
            'voice_settings': json.dumps({
                'stability': 0.5,
                'similarity_boost': 0.75
            }),
        },
    )
data = response.json()
```

**Responses**

- `202` Accepted - Voice changer task queued for processing

```json
{
  "task_id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "status": "pending"
}
```

### Get history list

`GET /v1/voice-changer/history`

Docs: https://vibi.pro/docs/voice-changer/get-history

Get list of voice changer tasks for the authenticated user

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page_size` | int | no | Number of tasks per page (max 100) Example: 30. |
| `page` | int | no | Page number (0-indexed) Example: 0. |

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/voice-changer/history?page_size=30&page=0" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/voice-changer/history?page_size=30&page=0', {
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const data = await response.json();
console.log(data.tasks);
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/voice-changer/history',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    params={'page_size': 30, 'page': 0}
)
history = response.json()
```

**Responses**

- `200` OK

```json
{
  "tasks": [
    {
      "id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
      "user_id": "user_123",
      "status": "completed",
      "progress": 100,
      "voice_id": "rv30Fd6w5bnbL0kHzWlr",
      "model_id": "eleven_multilingual_sts_v2",
      "source_audio_url": "https://api.vibi.pro/audio/vc_source_abc123.mp3",
      "result": {
        "audio_url": "https://api.vibi.pro/audio/vc_result_abc123.mp3"
      },
      "metadata": {
        "voice_settings": {
          "stability": 0.5,
          "similarity_boost": 0.75
        },
        "remove_noise": true
      },
      "credits_deducted": 500,
      "error": null,
      "detail_error": null,
      "created_at": "2026-03-17T10:30:00Z",
      "updated_at": "2026-03-17T10:30:15Z"
    }
  ],
  "has_more": false
}
```

### Get history detail

`GET /v1/voice-changer/history/{id}`

Docs: https://vibi.pro/docs/voice-changer/get-history-detail

Get details of a specific voice changer task by ID, including status and audio URL

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID |

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/voice-changer/history/{id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/voice-changer/history/{id}', {
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const task = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/voice-changer/history/{id}',
    headers={'xi-api-key': 'YOUR_API_KEY'}
)
task = response.json()
```

**Responses**

- `200` Status can be pending, processing, completed, or failed

```json
{
  "id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "user_id": "user_123",
  "status": "completed",
  "progress": 100,
  "voice_id": "rv30Fd6w5bnbL0kHzWlr",
  "model_id": "eleven_multilingual_sts_v2",
  "source_audio_url": "https://api.vibi.pro/audio/vc_source_abc123.mp3",
  "result": {
    "audio_url": "https://api.vibi.pro/audio/vc_result_abc123.mp3"
  },
  "metadata": {
    "voice_settings": {
      "stability": 0.5,
      "similarity_boost": 0.75
    },
    "remove_noise": false
  },
  "credits_deducted": 500,
  "error": null,
  "detail_error": null,
  "created_at": "2026-03-17T10:30:00Z",
  "updated_at": "2026-03-17T10:30:15Z"
}
```

### Retry task

`POST /v1/voice-changer/history/{id}/retry`

Docs: https://vibi.pro/docs/voice-changer/retry-task

Retry a failed or completed voice changer task. Re-uses the original source audio and re-deducts credits based on audio duration.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID to retry |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/voice-changer/history/{id}/retry" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/voice-changer/history/{id}/retry', {
  method: 'POST',
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/voice-changer/history/{id}/retry',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `202` Accepted - Task re-queued for processing

```json
{
  "task_id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "status": "pending"
}
```

### Delete task

`DELETE /v1/voice-changer/history/{id}`

Docs: https://vibi.pro/docs/voice-changer/delete-task

Delete a voice changer task and its associated audio files (source and result)

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID to delete |

**Example (curl)**

```bash
curl -X DELETE "https://api.vibi.pro/v1/voice-changer/history/{id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/voice-changer/history/{id}', {
  method: 'DELETE',
  headers: {
    'xi-api-key': 'YOUR_API_KEY'
  }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.delete(
    'https://api.vibi.pro/v1/voice-changer/history/{id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "message": "deleted"
}
```

## Dubbing

### 1. Init upload

`POST /v1/aidubbing/upload/init`

Docs: https://vibi.pro/docs/aidubbing/upload-init

Step 1 of creating an AI dubbing job
Opens an upload session and returns `upload_id` + recommended `chunk_size`
in bytes. Source max 2 GB, 2.5 hours (mp3, wav, aac, m4a, mp4, mov).
Flow: init -> chunk (xN) -> complete.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/aidubbing/upload/init" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/aidubbing/upload/init', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const { upload_id, chunk_size } = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/aidubbing/upload/init',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
session = response.json()
upload_id = session['upload_id']
chunk_size = session['chunk_size']
```

**Responses**

- `200` OK - upload session created

```json
{
  "upload_id": "0f8fad5b-d9cb-469f-a165-70867728950e",
  "chunk_size": 52428800
}
```

### 2. Upload chunks

`POST /v1/aidubbing/upload/chunk`

Docs: https://vibi.pro/docs/aidubbing/upload-chunk

Step 2: upload each file part at its 0-based index (a small file is a single
part). Any order, parallel OK; re-POST the same index to retry. Slice at
`chunk_size` from init; max 100 MB per part, 2 GB total per session.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Request body** (`multipart/form-data`, required)

| Field | Type | Required | Description |
|---|---|---|---|
| `upload_id` | string | yes | Session ID returned by upload/init. |
| `index` | int | yes | 0-based chunk index (max 511). |
| `chunk` | file | yes | The file slice for this index. Max 100 MB per part. |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/aidubbing/upload/chunk" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "upload_id=0f8fad5b-d9cb-469f-a165-70867728950e" \
  -F "index=0" \
  -F "chunk=@clip.part0"
```

**Example (JavaScript)**

```javascript
// Slice the file at chunk_size and POST each part
const chunkSize = 52428800; // from upload/init
const total = Math.ceil(file.size / chunkSize);

for (let i = 0; i < total; i++) {
  const formData = new FormData();
  formData.append('upload_id', uploadId);
  formData.append('index', String(i));
  formData.append('chunk', file.slice(i * chunkSize, (i + 1) * chunkSize));

  await fetch('https://api.vibi.pro/v1/aidubbing/upload/chunk', {
    method: 'POST',
    headers: { 'xi-api-key': 'YOUR_API_KEY' },
    body: formData
  });
}
```

**Example (Python)**

```python
import requests

CHUNK_SIZE = 52428800  # from upload/init

with open('clip.mp4', 'rb') as f:
    index = 0
    while True:
        part = f.read(CHUNK_SIZE)
        if not part:
            break
        requests.post(
            'https://api.vibi.pro/v1/aidubbing/upload/chunk',
            headers={'xi-api-key': 'YOUR_API_KEY'},
            data={'upload_id': upload_id, 'index': index},
            files={'chunk': ('part', part)},
        ).raise_for_status()
        index += 1
```

**Responses**

- `200` OK - chunk stored

```json
{
  "index": 0,
  "received": true
}
```

- `400` Bad request - invalid index, missing chunk data, chunk too large, or session over 2 GB

```json
{
  "error": "chunk too large"
}
```

- `404` Not found - unknown upload_id or session owned by another user

```json
{
  "error": "upload session not found"
}
```

### 3. Complete & start dubbing

`POST /v1/aidubbing/upload/complete`

Docs: https://vibi.pro/docs/aidubbing/upload-complete

Final step: reassembles all parts (0..total_chunks-1) and starts the dubbing
job with the parameters sent in this request. Costs 50 credits per second of
source duration, deducted upfront. Async - poll the history endpoint. The
upload session is consumed either way.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Request body** (`multipart/form-data`, required)

| Field | Type | Required | Description |
|---|---|---|---|
| `upload_id` | string | yes | Session ID returned by upload/init. |
| `total_chunks` | int | yes | Number of parts uploaded (1-512). All indexes 0..total_chunks-1 must be present. |
| `filename` | string | no | Original file name (used for display in history). Default: upload. |
| `content_type` | string | no | MIME type of the source file (e.g. video/mp4). Helps video detection. |
| `target_language` | string | yes | ISO 639-1 code to dub into (no locale suffix). Must be supported by the chosen provider. Supported: af, ar, hy, as, az, be, bn, bs, bg, ca, ceb, ny, zh, hr, cs, da, nl, en, et, fil, fi, fr, gl, ka, de, el, gu, ha, he, hi, hu, is, id, ga, it, ja, jv, kn, kk, ky, ko, lv, ln, lt, lb, mk, ms, ml, mr, ne, no, ps, fa, pl, pt, pa, ro, ru, sr, sd, sk, sl, so, es, sw, sv, ta, te, th, tr, uk, ur, vi, cy, nn, yue. |
| `source_language` | string | yes | Source language code. Required - `auto` NOT accepted. Same list as target plus mi, tl. |
| `provider` | string | no | `minimax`, `elevenlabs` or `capcut`. TTS model is pinned per provider (speech-2.8-hd / eleven_v3 / capcut). CapCut targets are limited to: en, zh, es, id, pt, fr, de, vi, ja, th, tr, it. Default: minimax. |
| `voice_id` | string | yes | Voice for the dubbed speech. MiniMax: official uniq_id (e.g. `Wise_Woman`) or your cloned voice ID. ElevenLabs: voice ID. CapCut: effect_id from the CapCut voice list - pick a voice whose language matches the target. |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/aidubbing/upload/complete" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "upload_id=0f8fad5b-d9cb-469f-a165-70867728950e" \
  -F "total_chunks=4" \
  -F "filename=clip.mp4" \
  -F "content_type=video/mp4" \
  -F "target_language=vi" \
  -F "source_language=en" \
  -F "provider=minimax" \
  -F "voice_id=Wise_Woman"
```

**Example (JavaScript)**

```javascript
const formData = new FormData();
formData.append('upload_id', uploadId);
formData.append('total_chunks', String(totalChunks));
formData.append('filename', file.name);
formData.append('content_type', file.type);
formData.append('target_language', 'vi');
formData.append('source_language', 'en');
formData.append('provider', 'minimax');
formData.append('voice_id', 'Wise_Woman');

const response = await fetch('https://api.vibi.pro/v1/aidubbing/upload/complete', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' },
  body: formData
});
const task = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/aidubbing/upload/complete',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    data={
        'upload_id': upload_id,
        'total_chunks': 4,
        'filename': 'clip.mp4',
        'content_type': 'video/mp4',
        'target_language': 'vi',
        'source_language': 'en',
        'provider': 'minimax',
        'voice_id': 'Wise_Woman',
    },
)
task = response.json()
```

**Responses**

- `202` Accepted - AI dubbing task queued for processing

```json
{
  "task_id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "status": "pending"
}
```

- `400` Bad request - missing chunk, invalid total_chunks, invalid dubbing params, or file too large

```json
{
  "error": "missing chunk 2"
}
```

- `402` Payment required - insufficient credits (50 credits per second of source)

```json
{
  "error": "insufficient credits",
  "required": 18000
}
```

- `404` Not found - unknown upload_id or session owned by another user

```json
{
  "error": "upload session not found"
}
```

### Get history list

`GET /v1/aidubbing/history`

Docs: https://vibi.pro/docs/aidubbing/get-history

Get list of AI dubbing tasks, ordered by newest first.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page_size` | int | no | Number of tasks per page (max 100) Example: 30. |
| `page` | int | no | Page number (0-indexed) Example: 0. |

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/aidubbing/history?page_size=30&page=0" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/aidubbing/history?page_size=30&page=0', {
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();
console.log(data.tasks);
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/aidubbing/history',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    params={'page_size': 30, 'page': 0},
)
history = response.json()
```

**Responses**

- `200` OK

```json
{
  "tasks": [
    {
      "id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
      "user_id": "user_123",
      "status": "completed",
      "progress": 100,
      "source_lang": "en",
      "target_lang": "vi",
      "provider": "minimax",
      "voice_id": "Wise_Woman",
      "model_id": "speech-2.8-hd",
      "source_audio_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e.mp3",
      "result": {
        "audio_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e.mp4",
        "srt_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e.target.srt"
      },
      "metadata": {
        "source_file_name": "interview.mp4",
        "original_video_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e_orig.mp4",
        "voice_name": "Wise Woman"
      },
      "credits_deducted": 18000,
      "created_at": "2026-06-11T10:30:00Z",
      "updated_at": "2026-06-11T10:35:15Z"
    }
  ],
  "has_more": false
}
```

### Get history detail

`GET /v1/aidubbing/history/{id}`

Docs: https://vibi.pro/docs/aidubbing/get-history-detail

Get one task by ID (use for polling). `result.audio_url` = dubbed video (mp4)
for video sources, else mp3; `result.srt_url` = translated subtitles.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID returned when the dubbing was created Example: 9b19d3e2-7c68-4eca-be48-e9917682d73e. |

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/aidubbing/history/{id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/aidubbing/history/{id}', {
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const task = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/aidubbing/history/{id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
task = response.json()
```

**Responses**

- `200` OK

```json
{
  "id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "user_id": "user_123",
  "status": "completed",
  "progress": 100,
  "source_lang": "en",
  "target_lang": "vi",
  "provider": "minimax",
  "voice_id": "Wise_Woman",
  "model_id": "speech-2.8-hd",
  "source_audio_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e.mp3",
  "result": {
    "audio_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e.mp4",
    "srt_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e.target.srt"
  },
  "metadata": {
    "source_file_name": "interview.mp4",
    "original_video_url": "https://api.vibi.pro/audio/9b19d3e2-7c68-4eca-be48-e9917682d73e_orig.mp4",
    "voice_name": "Wise Woman"
  },
  "credits_deducted": 18000,
  "created_at": "2026-06-11T10:30:00Z",
  "updated_at": "2026-06-11T10:35:15Z"
}
```

- `404` Not found - task does not exist or belongs to another user

```json
{
  "error": "task not found"
}
```

### Retry task

`POST /v1/aidubbing/history/{id}/retry`

Docs: https://vibi.pro/docs/aidubbing/retry-task

Re-run a completed or failed task from its stored source (no re-upload).
Re-deducts the original credit cost.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID to retry Example: 9b19d3e2-7c68-4eca-be48-e9917682d73e. |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/aidubbing/history/{id}/retry" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/aidubbing/history/{id}/retry', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/aidubbing/history/{id}/retry',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
task = response.json()
```

**Responses**

- `202` Accepted - task queued again

```json
{
  "task_id": "9b19d3e2-7c68-4eca-be48-e9917682d73e",
  "status": "pending"
}
```

- `402` Payment required - insufficient credits for the re-run

```json
{
  "error": "insufficient credits",
  "required": 18000
}
```

- `404` Not found - task does not exist or belongs to another user

```json
{
  "error": "task not found"
}
```

- `409` Conflict - task is still pending/processing

```json
{
  "error": "only completed or failed tasks can be retried"
}
```

- `422` Unprocessable - source file no longer available (credits refunded)

```json
{
  "error": "cannot retry: source file no longer available"
}
```

### Delete task

`DELETE /v1/aidubbing/history/{id}`

Docs: https://vibi.pro/docs/aidubbing/delete-task

Delete an AI dubbing task and its stored files.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Task ID to delete Example: 9b19d3e2-7c68-4eca-be48-e9917682d73e. |

**Example (curl)**

```bash
curl -X DELETE "https://api.vibi.pro/v1/aidubbing/history/{id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/aidubbing/history/{id}', {
  method: 'DELETE',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.delete(
    'https://api.vibi.pro/v1/aidubbing/history/{id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
result = response.json()
```

**Responses**

- `200` OK - task deleted

```json
{
  "status": "deleted"
}
```

- `404` Not found - task does not exist or belongs to another user

```json
{
  "error": "task not found"
}
```

## Speech to text

### Speech to text

`POST /v1/speech-to-text`

Docs: https://vibi.pro/docs/stt/speech-to-text

Convert speech audio to text using ElevenLabs Scribe v2 model.
All optional parameters fall back to ElevenLabs defaults when omitted.

**This endpoint is asynchronous.** It returns `202 Accepted` with a `task_id`
immediately; the transcript is fetched by polling
`GET /v1/speech-to-text/{task_id}` until `status` is `completed`.

Credits are charged from the audio duration at submit time, so the exact cost
is known before the work starts, and are refunded in full if the job fails.

`language_code` is **required**.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Request body** (`multipart/form-data`, required)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file (binary) | yes | Audio/video file to transcribe. Max 90MB. Supported formats: AAC, AIFF, OGG, MP3, Opus, WAV, WebM, FLAC, M4A. |
| `language_code` | string | yes | ISO-639-1 or ISO-639-3 language code (e.g. "en", "eng", "vi", "vie"). **Required.** A request without it is rejected with 400 `missing_language_code`, and an unrecognised code with 400 `invalid_language_code` - neither consumes credits. |
| `timestamps_granularity` | string | no | Granularity level for timestamps in the response. - `none` - no timestamps - `word` - timestamp per word (default) - `character` - timestamp per character Default: word. |
| `diarize` | boolean | no | Enable speaker diarization - annotates each word with a `speaker_id` ("speaker_0", "speaker_1", ...). Use `num_speakers` for better accuracy when the number of speakers is known. Default: false. |
| `num_speakers` | integer | no | Hint for the maximum number of speakers present (1-32). When provided, improves diarization accuracy. Only relevant when `diarize=true`. Range: 1 to 32. |
| `diarization_threshold` | number | no | Threshold for speaker separation (0.0-1.5). Default: 0.22. Higher values reduce the chance of one speaker being split into two, but increase the chance of merging two different speakers. Only applies when `diarize=true` and `num_speakers` is not set. Range: 0 to 1.5. |
| `tag_audio_events` | boolean | no | Include non-speech audio event tags in the transcript, e.g. (laughter), (footsteps), (applause). Defaults to `true` - set to `false` to suppress them. Default: true. |
| `no_verbatim` | boolean | no | Remove filler words and false starts ("um", "uh", "like", restarted sentences) for a cleaner transcript. Only supported on `scribe_v2`. Default: false. |
| `temperature` | number | no | Sampling temperature (0.0-2.0). Higher values produce more diverse output; lower values are more deterministic. Default is model-specific. Range: 0 to 2. |
| `seed` | integer | no | Integer seed (0-2147483647) for deterministic output. Same seed + same input produces the same transcript. Range: 0 to 2147483647. |
| `file_format` | string | no | Input audio format hint. Use `pcm_s16le_16` for raw 16-bit PCM at 16kHz; `other` for all other formats. Default: other. |
| `use_multi_channel` | boolean | no | Transcribe each audio channel independently (up to 5 channels). Each channel is assigned a `speaker_id` based on its channel number. Cannot be used together with `diarize=true`. Default: false. |
| `keyterms` | array of strings | no | List of terms to bias the model toward (e.g. product names, technical jargon, proper nouns). Max 1000 terms, each under 50 characters. Send as repeated form fields: `keyterms=Term1&keyterms=Term2`. |
| `entity_detection` | array of strings | no | Detect named entities in the transcript. Pass `all` to detect everything, or specify categories/types. Categories: `pii`, `phi`, `pci`, `other`, `offensive_language`. Send as repeated form fields: `entity_detection=pii&entity_detection=phi`. |
| `entity_redaction` | array of strings | no | Redact detected entities from the transcript. Must be a subset of `entity_detection`. Send as repeated form fields: `entity_redaction=pii`. |
| `entity_redaction_mode` | string | no | Format used to replace redacted entities in the transcript text. - `redacted` - replaced with `[redacted]` - `entity_type` - replaced with the entity type, e.g. `[PERSON_NAME]` - `enumerated_entity_type` - replaced with type + index, e.g. `[PERSON_NAME_1]` Default: enumerated_entity_type. |
| `additional_formats` | array of strings | no | Request the transcript in additional export formats alongside the JSON response. The content is returned in the `additional_formats` array in the response. Send as repeated form fields: `additional_formats=srt&additional_formats=txt`. |

**Example (curl)**

```bash
# 1. Submit - returns {"task_id": "...", "status": "pending"}
curl -X POST "https://api.vibi.pro/v1/speech-to-text" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "file=@recording.mp3"

# 2. Poll until status is "completed", then read .result
curl "https://api.vibi.pro/v1/speech-to-text/TASK_ID" \
  -H "xi-api-key: YOUR_API_KEY"

# List your transcriptions (without the transcripts)
curl "https://api.vibi.pro/v1/speech-to-text?page_size=20" \
  -H "xi-api-key: YOUR_API_KEY"

# With speaker diarization (identify who speaks when)
curl -X POST "https://api.vibi.pro/v1/speech-to-text" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "file=@interview.mp3" \
  -F "diarize=true" \
  -F "num_speakers=2"

# Clean transcript with SRT export, no filler words
curl -X POST "https://api.vibi.pro/v1/speech-to-text" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "file=@recording.mp3" \
  -F "no_verbatim=true" \
  -F "additional_formats=srt"

# With keyterms and language hint
curl -X POST "https://api.vibi.pro/v1/speech-to-text" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "file=@recording.mp3" \
  -F "language_code=en" \
  -F "keyterms=Acme Corp" \
  -F "keyterms=Q4 Report"
```

**Example (JavaScript)**

```javascript
// Basic transcription
const formData = new FormData();
formData.append('file', audioBlob, 'recording.mp3');

// Submit, then poll until the transcript is ready.
const submit = await fetch('https://api.vibi.pro/v1/speech-to-text', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' },
  body: formData
});
const { task_id } = await submit.json();   // 202 Accepted

let task;
do {
  await new Promise(r => setTimeout(r, 2000));
  const poll = await fetch(`https://api.vibi.pro/v1/speech-to-text/${task_id}`, {
    headers: { 'xi-api-key': 'YOUR_API_KEY' }
  });
  task = await poll.json();
} while (task.status === 'pending' || task.status === 'processing');

if (task.status === 'failed') throw new Error(task.detail_error);
console.log(task.result.text);

// With speaker diarization
const form = new FormData();
form.append('file', audioBlob, 'interview.mp3');
form.append('diarize', 'true');
form.append('num_speakers', '2');

const res = await fetch('https://api.vibi.pro/v1/speech-to-text', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' },
  body: form
});
const result = await res.json();

// Group transcript by speaker
const bySpeaker = {};
for (const word of result.words) {
  if (word.type === 'word') {
    const spk = word.speaker_id ?? 'unknown';
    (bySpeaker[spk] ??= []).push(word.text);
  }
}
for (const [spk, words] of Object.entries(bySpeaker)) {
  console.log(`${spk}: ${words.join(' ')}`);
}

// With keyterms + SRT export
const f = new FormData();
f.append('file', audioBlob, 'recording.mp3');
['Acme Corp', 'Q4 Report'].forEach(t => f.append('keyterms', t));
f.append('no_verbatim', 'true');
f.append('additional_formats', 'srt');

const r = await fetch('https://api.vibi.pro/v1/speech-to-text', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' },
  body: f
});
const out = await r.json();
const srt = out.additional_formats?.find(x => x.file_extension === 'srt');
if (srt) console.log(srt.content);
```

**Example (Python)**

```python
import requests
import time

# Basic transcription
# Submit, then poll until the transcript is ready.
submit = requests.post(
    'https://api.vibi.pro/v1/speech-to-text',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    files={'file': open('recording.mp3', 'rb')},
)
task_id = submit.json()['task_id']          # 202 Accepted

while True:
    task = requests.get(
        f'https://api.vibi.pro/v1/speech-to-text/{task_id}',
        headers={'xi-api-key': 'YOUR_API_KEY'},
    ).json()
    if task['status'] in ('completed', 'failed'):
        break
    time.sleep(2)

if task['status'] == 'failed':
    raise RuntimeError(task['detail_error'])
print(task['result']['text'])

# With speaker diarization
response = requests.post(
    'https://api.vibi.pro/v1/speech-to-text',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    files={'file': open('interview.mp3', 'rb')},
    data={
        'diarize': 'true',
        'num_speakers': '2',
    }
)
data = response.json()

# Group transcript by speaker
by_speaker = {}
for word in data['words']:
    if word['type'] == 'word':
        spk = word.get('speaker_id', 'unknown')
        by_speaker.setdefault(spk, []).append(word['text'])
for spk, words in by_speaker.items():
    print(f"{spk}: {' '.join(words)}")

# With keyterms + SRT export
response = requests.post(
    'https://api.vibi.pro/v1/speech-to-text',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    files={'file': open('recording.mp3', 'rb')},
    data=[
        ('no_verbatim', 'true'),
        ('keyterms', 'Acme Corp'),
        ('keyterms', 'Q4 Report'),
        ('additional_formats', 'srt'),
    ]
)
data = response.json()
for fmt in data.get('additional_formats', []):
    if fmt['file_extension'] == 'srt':
        print(fmt['content'])
```

**Responses**

- `202` Transcription accepted and queued. Poll `GET /v1/speech-to-text/{task_id}` for the result.

```json
{
  "task_id": "f534211b-b414-4697-bb58-a2a7a688a49f",
  "status": "pending"
}
```

- `200` Returned by `GET /v1/speech-to-text/{task_id}`. `status` is one of `pending`, `processing`, `completed`, `failed`. `result` is present only once `status` is `completed`, and holds the transcript described below. On `failed`, `detail_error` explains why and the credits have already been returned. Key response fields: - `language_code`: detected language in ISO-639-3 (e.g. "eng", "vie"). - `language_probability`: 0.0-1.0 confidence in language detection. - `text`: full transcript as a plain string. - `transcription_id`: unique ID for this transcription job. - `words[].type`: `"word"` (spoken), `"spacing"` (gap), `"audio_event"` (non-speech). - `words[].start` / `words[].end`: timestamp in seconds for each token. - `words[].logprob`: log-probability confidence per word (0 = 100%, -0.5 ~ 61%, -1.0 ~ 37%). - `words[].speaker_id`: present when `diarize=true` - e.g. `"speaker_0"`, `"speaker_1"`. - `additional_formats[].file_extension`: format identifier (e.g. `"srt"`, `"txt"`). - `additional_formats[].content`: transcript in the requested format (SRT, TXT, etc.).

```json
{
  "language_code": "eng",
  "language_probability": 0.9746,
  "text": "Hello everyone. My name is John. Thanks for joining.",
  "words": [
    {
      "text": "Hello",
      "start": 0.0,
      "end": 0.32,
      "type": "word",
      "logprob": -0.002,
      "speaker_id": "speaker_0"
    },
    {
      "text": " ",
      "start": 0.32,
      "end": 0.4,
      "type": "spacing",
      "logprob": -0.001,
      "speaker_id": "speaker_0"
    },
    {
      "text": "everyone.",
      "start": 0.4,
      "end": 0.92,
      "type": "word",
      "logprob": -0.04,
      "speaker_id": "speaker_0"
    },
    {
      "text": "My",
      "start": 1.1,
      "end": 1.22,
      "type": "word",
      "logprob": -0.0001,
      "speaker_id": "speaker_0"
    },
    {
      "text": "name",
      "start": 1.22,
      "end": 1.36,
      "type": "word",
      "logprob": -0.0001,
      "speaker_id": "speaker_0"
    },
    {
      "text": "is",
      "start": 1.36,
      "end": 1.44,
      "type": "word",
      "logprob": -0.0008,
      "speaker_id": "speaker_0"
    },
    {
      "text": "John.",
      "start": 1.44,
      "end": 1.98,
      "type": "word",
      "logprob": -0.03,
      "speaker_id": "speaker_0"
    },
    {
      "text": "Thanks",
      "start": 3.1,
      "end": 3.4,
      "type": "word",
      "logprob": -0.01,
      "speaker_id": "speaker_1"
    },
    {
      "text": "for",
      "start": 3.4,
      "end": 3.5,
      "type": "word",
      "logprob": -0.005,
      "speaker_id": "speaker_1"
    },
    {
      "text": "joining.",
      "start": 3.5,
      "end": 4.1,
      "type": "word",
      "logprob": -0.008,
      "speaker_id": "speaker_1"
    }
  ],
  "additional_formats": [],
  "transcription_id": "Qe71uRUkuArC2JNlRsGy"
}
```

### Get transcription

`GET /v1/speech-to-text/{task_id}`

Docs: https://vibi.pro/docs/stt/get-transcription

Poll a transcription submitted with `POST /v1/speech-to-text`.

Poll until `status` is `completed` or `failed`. Two seconds between polls is a
sensible interval. On `completed`, `result` holds the transcript; on `failed`,
`detail_error` explains why and the credits have already been refunded in full.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `task_id` | string | yes | The `task_id` returned when the audio was submitted. |

**Example (curl)**

```bash
curl "https://api.vibi.pro/v1/speech-to-text/{task_id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
// Poll until the transcript is ready.
let task;
do {
  await new Promise(r => setTimeout(r, 2000));
  const res = await fetch(`https://api.vibi.pro/v1/speech-to-text/${taskId}`, {
    headers: { 'xi-api-key': 'YOUR_API_KEY' }
  });
  task = await res.json();
} while (task.status === 'pending' || task.status === 'processing');

if (task.status === 'failed') throw new Error(task.detail_error);
console.log(task.result.text);
```

**Example (Python)**

```python
import requests
import time

while True:
    task = requests.get(
        f'https://api.vibi.pro/v1/speech-to-text/{task_id}',
        headers={'xi-api-key': 'YOUR_API_KEY'},
    ).json()
    if task['status'] in ('completed', 'failed'):
        break
    time.sleep(2)

if task['status'] == 'failed':
    raise RuntimeError(task['detail_error'])
print(task['result']['text'])
```

**Responses**

- `200` `status` is one of `pending`, `processing`, `completed`, `failed`. `result` appears only once the transcription has completed, and matches the response documented on the submit endpoint.

```json
{
  "task_id": "f534211b-b414-4697-bb58-a2a7a688a49f",
  "status": "completed",
  "progress": 100,
  "filename": "recording.mp3",
  "audio_duration_secs": 56.145875,
  "credits_deducted": 17,
  "created_at": "2026-09-07T13:50:48Z",
  "updated_at": "2026-09-07T13:50:56Z",
  "result": {
    "language_code": "eng",
    "language_probability": 1.0,
    "text": "Good morning, everyone...",
    "words": [
      { "text": "Good", "start": 0.119, "end": 0.319, "type": "word", "logprob": -0.002 }
    ],
    "transcription_id": "UhPzPRMAP6TIfBfx8BHo",
    "audio_duration_secs": 56.145875
  }
}
```

- `404` No transcription with that id belongs to your account.

```json
{ "error": "task not found" }
```

### List transcriptions

`GET /v1/speech-to-text`

Docs: https://vibi.pro/docs/stt/list-transcriptions

Your transcriptions, newest first.

Transcripts are NOT included - a page of them would be megabytes. Fetch a
single task by id to read its `result`.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page` | integer | no | Zero-based page number. Default: 0. |
| `page_size` | integer | no | Items per page, maximum 100. Default: 30. |

**Example (curl)**

```bash
curl "https://api.vibi.pro/v1/speech-to-text?page=0&page_size=20" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const res = await fetch('https://api.vibi.pro/v1/speech-to-text?page_size=20', {
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const { tasks, has_more } = await res.json();
for (const t of tasks) console.log(t.task_id, t.status, t.filename);
```

**Example (Python)**

```python
import requests

data = requests.get(
    'https://api.vibi.pro/v1/speech-to-text',
    headers={'xi-api-key': 'YOUR_API_KEY'},
    params={'page_size': 20},
).json()
for t in data['tasks']:
    print(t['task_id'], t['status'], t.get('filename'))
```

**Responses**

- `200` A page of transcriptions, without their transcripts.

```json
{
  "tasks": [
    {
      "task_id": "f534211b-b414-4697-bb58-a2a7a688a49f",
      "status": "completed",
      "progress": 100,
      "filename": "recording.mp3",
      "audio_duration_secs": 56.145875,
      "credits_deducted": 17,
      "created_at": "2026-09-07T13:50:48Z"
    }
  ],
  "has_more": false
}
```

### Delete transcription

`DELETE /v1/speech-to-text/{task_id}`

Docs: https://vibi.pro/docs/stt/delete-transcription

Delete a transcription and the audio stored with it. Permanent.

Only finished transcriptions can be deleted. A job that is still `pending` or
`processing` returns 409 - wait for it to finish first. Deleting does not
refund credits; a completed transcription has already been delivered.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `task_id` | string | yes | The transcription to delete. |

**Example (curl)**

```bash
curl -X DELETE "https://api.vibi.pro/v1/speech-to-text/{task_id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
await fetch(`https://api.vibi.pro/v1/speech-to-text/${taskId}`, {
  method: 'DELETE',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
```

**Example (Python)**

```python
import requests

requests.delete(
    f'https://api.vibi.pro/v1/speech-to-text/{task_id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
```

**Responses**

- `200` Deleted.

```json
{ "deleted": true }
```

- `409` The transcription is still running and cannot be deleted yet.

```json
{
  "error": "task_still_running",
  "message": "This transcription is still running. Wait for it to finish before deleting it."
}
```

- `404` No transcription with that id belongs to your account.

```json
{ "error": "task not found" }
```

## Models

### Get models

`GET /v1/models`

Docs: https://vibi.pro/docs/models/get-models

Get all available TTS models. Defaults to ElevenLabs models. Pass provider=minimax for MiniMax or provider=capcut for CapCut models.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `provider` | string | no | Provider to get models for. Defaults to ElevenLabs. Use "minimax" for MiniMax or "capcut" for CapCut models. |

**Example (curl)**

```bash
# ElevenLabs models (default)
curl -X GET "https://api.vibi.pro/v1/models" \
  -H "xi-api-key: YOUR_API_KEY"

# MiniMax models
curl -X GET "https://api.vibi.pro/v1/models?provider=minimax" \
  -H "xi-api-key: YOUR_API_KEY"

# CapCut models
curl -X GET "https://api.vibi.pro/v1/models?provider=capcut" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
// ElevenLabs models (default)
const response = await fetch('https://api.vibi.pro/v1/models', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();

// MiniMax models
const mmResponse = await fetch('https://api.vibi.pro/v1/models?provider=minimax', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const mmData = await mmResponse.json();

// CapCut models
const ccResponse = await fetch('https://api.vibi.pro/v1/models?provider=capcut', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const ccData = await ccResponse.json();
```

**Example (Python)**

```python
import requests

headers = {'xi-api-key': 'YOUR_API_KEY'}

# ElevenLabs models (default)
response = requests.get('https://api.vibi.pro/v1/models', headers=headers)
data = response.json()

# MiniMax models
response = requests.get('https://api.vibi.pro/v1/models', params={'provider': 'minimax'}, headers=headers)
data = response.json()

# CapCut models
response = requests.get('https://api.vibi.pro/v1/models', params={'provider': 'capcut'}, headers=headers)
data = response.json()
```

**Responses**

- `200` ElevenLabs models (default)

```json
[
  {
    "model_id": "eleven_v3",
    "name": "Eleven v3",
    "description": "Our highly expressive model. Supports 70+ languages. Requires more prompt engineering than our previous models.",
    "can_do_text_to_speech": true,
    "can_do_voice_conversion": false,
    "can_use_style": false,
    "can_use_speaker_boost": false,
    "can_be_finetuned": false,
    "serves_pro_voices": false,
    "requires_alpha_access": false,
    "max_characters_request_free_user": 5000,
    "max_characters_request_subscribed_user": 5000,
    "maximum_text_length_per_request": 5000,
    "token_cost_factor": 1,
    "concurrency_group": "standard",
    "model_rates": {
      "character_cost_multiplier": 1
    },
    "languages": [
      {
          "language_id": "en",
          "name": "English"
      }
    ],
    "supports_context": false,
    "supports_speed": true,
    "supports_similarity": false
  }
]
```

- `200` MiniMax models (provider=minimax)

```json
[
  {
    "model_id": "speech-2.8-hd",
    "name": "Speech 2.8 HD",
    "description": "Latest generation with highest quality audio. Best for professional productions.",
    "img": "https://cdn.hailuoai.video/...",
    "badge": "newest",
    "credit_ratio": 1
  }
]
```

- `200` CapCut models (provider=capcut). credit_ratio is per-character rate (0 = free).

```json
[
  {
    "model_id": "capcut",
    "name": "CapCut",
    "description": "CapCut engine TTS",
    "credit_ratio": 1,
    "languages": ["en", "vi", "zh", "id", "es", "pt", "ja", "th"]
  }
]
```

## Languages

### Get languages

`GET /v1/languages`

Docs: https://vibi.pro/docs/languages/get-languages

Get supported languages for a provider. Defaults to ElevenLabs. Pass provider=minimax for MiniMax or provider=capcut for CapCut. ElevenLabs and CapCut return ISO codes (e.g. "vi"), MiniMax returns full names (e.g. "Vietnamese"). The backend normalizes both formats automatically in TTS requests.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `provider` | string | no | Provider to get languages for. Defaults to ElevenLabs. Use "minimax" for MiniMax or "capcut" for CapCut languages. |

**Example (curl)**

```bash
# ElevenLabs languages (default)
curl -X GET "https://api.vibi.pro/v1/languages" \
  -H "xi-api-key: YOUR_API_KEY"

# MiniMax languages
curl -X GET "https://api.vibi.pro/v1/languages?provider=minimax" \
  -H "xi-api-key: YOUR_API_KEY"

# CapCut languages
curl -X GET "https://api.vibi.pro/v1/languages?provider=capcut" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
// ElevenLabs languages (default)
const response = await fetch('https://api.vibi.pro/v1/languages', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const languages = await response.json();

// MiniMax languages
const mmResponse = await fetch('https://api.vibi.pro/v1/languages?provider=minimax', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const mmLanguages = await mmResponse.json();

// CapCut languages
const ccResponse = await fetch('https://api.vibi.pro/v1/languages?provider=capcut', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const ccLanguages = await ccResponse.json();
```

**Example (Python)**

```python
import requests

headers = {'xi-api-key': 'YOUR_API_KEY'}

# ElevenLabs languages (default)
response = requests.get('https://api.vibi.pro/v1/languages', headers=headers)
languages = response.json()

# MiniMax languages
response = requests.get('https://api.vibi.pro/v1/languages', params={'provider': 'minimax'}, headers=headers)
languages = response.json()

# CapCut languages
response = requests.get('https://api.vibi.pro/v1/languages', params={'provider': 'capcut'}, headers=headers)
languages = response.json()
```

**Responses**

- `200` ElevenLabs languages (default) - ISO codes

```json
[
  { "code": "af", "name": "Afrikaans" },
  { "code": "ar", "name": "Arabic" },
  { "code": "en", "name": "English" },
  { "code": "vi", "name": "Vietnamese" },
  { "code": "zh", "name": "Mandarin Chinese" }
]
```

- `200` MiniMax languages (provider=minimax) - full names

```json
[
  { "code": "Afrikaans", "name": "Afrikaans" },
  { "code": "Arabic", "name": "Arabic" },
  { "code": "English", "name": "English" },
  { "code": "Vietnamese", "name": "Vietnamese" },
  { "code": "Chinese (Mandarin)", "name": "Chinese (Mandarin)" }
]
```

- `200` CapCut languages (provider=capcut) - ISO codes

```json
[
  { "code": "zh", "name": "Chinese" },
  { "code": "en", "name": "English" },
  { "code": "id", "name": "Indonesian" },
  { "code": "ja", "name": "Japanese" },
  { "code": "pt", "name": "Portuguese" },
  { "code": "es", "name": "Spanish" },
  { "code": "th", "name": "Thai" },
  { "code": "vi", "name": "Vietnamese" }
]
```

## Elevenlabs Voices

### List shared voices

`GET /v1/shared-voices`

Docs: https://vibi.pro/docs/voices/list-shared-voices

Browse the ElevenLabs shared voice library with filters, pagination, and sorting. Supports both cursor-based and page-based pagination.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page_size` | integer | no | Number of voices per page (max 100) |
| `page` | integer | no | Page number (0-indexed). Used for page-based pagination. |
| `search` | string | no | Search term to filter voices by name or description |
| `sort` | string | no | Sort order: trending, created_date, cloned_by_count, usage_character_count_1y |
| `category` | string | no | Quality filter: high_quality (or omit for all) |
| `gender` | string | no | Gender filter: male, female, neutral (or omit for all) |
| `age` | string | no | Age filter: young, middle_aged, old (or omit for all) |
| `accent` | string | no | Accent filter (e.g. american, british, australian). Omit for all |
| `required_languages` | string | no | Language code filter (e.g. en, es, fr) |
| `use_cases[]` | array | no | Use case filter (array). Values: conversational, narrative_and_story, characters_and_animation, social_media, entertainment_and_tv, advertisement, informative_and_educational. Omit for all |

**Example (curl)**

```bash
# Basic request (trending voices)
curl -X GET "https://api.vibi.pro/v1/shared-voices?page_size=30&sort=trending" \
  -H "xi-api-key: YOUR_API_KEY"

# With filters
curl -X GET "https://api.vibi.pro/v1/shared-voices\
  ?page_size=30\
  &gender=male\
  &age=young\
  &accent=British\
  &required_languages=en\
  &category=high_quality\
  &use_cases%5B%5D=entertainment_and_tv\
  &use_cases%5B%5D=narrative_and_story\
  &sort=cloned_by_count\
  &page=0" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
// Basic request (trending voices)
const response = await fetch('https://api.vibi.pro/v1/shared-voices?page_size=30&sort=trending', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();

// With filters
const params = new URLSearchParams({
  page_size: '30',
  gender: 'male',
  age: 'young',
  accent: 'British',
  required_languages: 'en',
  category: 'high_quality',
  sort: 'cloned_by_count',
  page: '0'
});
params.append('use_cases[]', 'entertainment_and_tv');
params.append('use_cases[]', 'narrative_and_story');

const filtered = await fetch(`https://api.vibi.pro/v1/shared-voices?${params}`, {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const filteredData = await filtered.json();
```

**Example (Python)**

```python
import requests

headers = {'xi-api-key': 'YOUR_API_KEY'}

# Basic request (trending voices)
response = requests.get(
    'https://api.vibi.pro/v1/shared-voices',
    headers=headers,
    params={'page_size': 30, 'sort': 'trending'}
)
data = response.json()

# With filters
response = requests.get(
    'https://api.vibi.pro/v1/shared-voices',
    headers=headers,
    params={
        'page_size': 30,
        'gender': 'male',
        'age': 'young',
        'accent': 'British',
        'required_languages': 'en',
        'category': 'high_quality',
        'use_cases[]': ['entertainment_and_tv', 'narrative_and_story'],
        'sort': 'cloned_by_count',
        'page': 0
    }
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "voices": [
    {
      "voice_id": "pNInz6obpgDQGcFmaJgB",
      "name": "Adam",
      "description": "A warm, friendly male voice with an American accent.",
      "preview_url": "https://storage.googleapis.com/eleven-public-prod/audio/adam.mp3",
      "category": "professional",
      "gender": "male",
      "age": "middle_aged",
      "accent": "american",
      "language": "en",
      "use_case": "narration",
      "descriptive": "warm",
      "free_users_allowed": true,
      "usage_character_count_1y": 1500000,
      "usage_character_count_7d": 25000,
      "cloned_by_count": 1200,
      "featured": false,
      "rate": 0,
      "live_moderation_enabled": false,
      "date_unix": 1700000000,
      "verified_languages": [
        {
          "language_id": "en",
          "language": "English",
          "model_id": "eleven_multilingual_v2",
          "accent": "American",
          "locale": "en-US",
          "preview_url": "https://storage.googleapis.com/audio/adam-en.mp3"
        }
      ],
      "image_url": "https://storage.googleapis.com/images/adam.jpg"
    }
  ],
  "has_more": true,
  "last_sort_id": "abc123def456"
}
```

### List default voices

`GET /v1/default-voices`

Docs: https://vibi.pro/docs/voices/list-default-voices

Get ElevenLabs default/premade voices. These are high-quality built-in voices available to all users.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page_size` | integer | no | Number of voices per page |
| `search` | string | no | Search term to filter voices by name |

**Example (curl)**

```bash
# Get default voices
curl -X GET "https://api.vibi.pro/v1/default-voices?page_size=30" \
  -H "xi-api-key: YOUR_API_KEY"

# Search for a specific voice
curl -X GET "https://api.vibi.pro/v1/default-voices?search=aria&page_size=30" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
// Get default voices
const response = await fetch('https://api.vibi.pro/v1/default-voices?page_size=30', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();

// Search for a specific voice
const searchResponse = await fetch('https://api.vibi.pro/v1/default-voices?search=aria&page_size=30', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const searchData = await searchResponse.json();
```

**Example (Python)**

```python
import requests

headers = {'xi-api-key': 'YOUR_API_KEY'}

# Get default voices
response = requests.get(
    'https://api.vibi.pro/v1/default-voices',
    headers=headers,
    params={'page_size': 30}
)
data = response.json()

# Search for a specific voice
response = requests.get(
    'https://api.vibi.pro/v1/default-voices',
    headers=headers,
    params={'search': 'aria', 'page_size': 30}
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "voices": [
    {
      "voice_id": "rv30Fd6w5bnbL0kHzWlr",
      "name": "Aria",
      "description": "A young, expressive female voice with a clear American accent.",
      "preview_url": "https://storage.googleapis.com/eleven-public-prod/audio/aria.mp3",
      "category": "premade",
      "gender": "female",
      "age": "young",
      "accent": "american",
      "language": "en",
      "use_case": "social_media",
      "descriptive": "expressive",
      "free_users_allowed": true,
      "verified_languages": [
        {
          "language_id": "en",
          "language": "English",
          "model_id": "eleven_multilingual_v2",
          "accent": "American",
          "locale": "en-US"
        }
      ]
    },
    {
      "voice_id": "CwhRBWXzGAHq8TQ4Fs17",
      "name": "Roger",
      "description": "A confident, mature male voice with a British accent.",
      "preview_url": "https://storage.googleapis.com/eleven-public-prod/audio/roger.mp3",
      "category": "premade",
      "gender": "male",
      "age": "middle_aged",
      "accent": "british",
      "language": "en",
      "use_case": "narration",
      "descriptive": "confident",
      "free_users_allowed": true,
      "verified_languages": [
        {
          "language_id": "en",
          "language": "English",
          "model_id": "eleven_multilingual_v2",
          "accent": "British"
        }
      ]
    }
  ],
  "has_more": true,
  "last_sort_id": "next_page_token_value"
}
```

## MiniMax Voices

### List system voices

`GET /v1/minimax/system-voices`

Docs: https://vibi.pro/docs/minimax-voices/list-system-voices

Browse MiniMax system voice library with optional filters.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page` | integer | no | Page number (1-indexed) |
| `page_size` | integer | no | Number of voices per page |
| `search` | string | no | Search text to filter voices by name |
| `gender` | string | no | Gender filter (e.g. Male, Female) |
| `language` | string | no | Language filter (e.g. English, Chinese (Mandarin), Japanese) |
| `accent` | string | no | Accent filter (e.g. EN-US (General), EN-British, EN-Australian) |
| `age` | string | no | Age filter, comma-separated for multiple (e.g. Young, Middle-aged) |
| `use_cases` | string | no | Use case filter, comma-separated for multiple (e.g. Audiobook,Podcast) |

**Example (curl)**

```bash
# Basic request (first page)
curl -X GET "https://api.vibi.pro/v1/minimax/system-voices?page=1&page_size=30" \
  -H "xi-api-key: YOUR_API_KEY"

# With filters
curl -X GET "https://api.vibi.pro/v1/minimax/system-voices\
  ?page=1\
  &page_size=30\
  &language=English\
  &gender=Male\
  &age=Young\
  &accent=EN-US%20(General)\
  &use_cases=Audiobook,Podcast\
  &search=narrator" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
// Basic request (first page)
const response = await fetch('https://api.vibi.pro/v1/minimax/system-voices?page=1&page_size=30', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();

// With filters
const params = new URLSearchParams({
  page: '1',
  page_size: '30',
  language: 'English',
  gender: 'Male',
  age: 'Young',
  accent: 'EN-US (General)',
  use_cases: 'Audiobook,Podcast',
  search: 'narrator',
});
const filtered = await fetch(`https://api.vibi.pro/v1/minimax/system-voices?${params}`, {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const filteredData = await filtered.json();
```

**Example (Python)**

```python
import requests

headers = {'xi-api-key': 'YOUR_API_KEY'}

# Basic request (first page)
response = requests.get(
    'https://api.vibi.pro/v1/minimax/system-voices',
    headers=headers,
    params={'page': 1, 'page_size': 30},
)
data = response.json()

# With filters
response = requests.get(
    'https://api.vibi.pro/v1/minimax/system-voices',
    headers=headers,
    params={
        'page': 1,
        'page_size': 30,
        'language': 'English',
        'gender': 'Male',
        'age': 'Young',
        'accent': 'EN-US (General)',
        'use_cases': 'Audiobook,Podcast',
        'search': 'narrator',
    },
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "voice_list": [
    {
      "voice_id": "226905123659939",
      "voice_name": "Aussie Bloke",
      "uniq_id": "English_Aussie_Bloke",
      "tag_list": ["English", "Male", "Young", "EN-Australian"],
      "cover_url": "https://cdn.hailuoai.video/moss/voice_cover/xxx.webp",
      "sample_audio": "https://cdn.hailuoai.video/moss/voice_sample/xxx.mp3",
      "description": "A warm Australian male voice, great for storytelling and narration."
    },
    {
      "voice_id": "226905123659940",
      "voice_name": "British Lady",
      "uniq_id": "English_British_Lady",
      "tag_list": ["English", "Female", "Middle-aged", "EN-British"],
      "cover_url": "https://cdn.hailuoai.video/moss/voice_cover/yyy.webp",
      "sample_audio": "https://cdn.hailuoai.video/moss/voice_sample/yyy.mp3",
      "description": "A refined British female voice, ideal for audiobooks and podcasts."
    }
  ],
  "total": 150,
  "has_more": true
}
```

### List cloned voices

`GET /v1/minimax/voices`

Docs: https://vibi.pro/docs/minimax-voices/list-cloned-voices

Get all cloned voices for the authenticated user. Returns voices with status processing, done, or failed (excludes deleted).

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Example (curl)**

```bash
curl -X GET "https://api.vibi.pro/v1/minimax/voices" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/minimax/voices', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.get(
    'https://api.vibi.pro/v1/minimax/voices',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "voices": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "voice_name": "My Custom Voice",
      "sample_audio_url": "https://api.vibi.pro/audio/minimax/user123/a1b2c3d4_sample.mp3",
      "cover_url": "https://api.vibi.pro/audio/minimax/user123/a1b2c3d4_cover.png",
      "language_tag": "English",
      "gender": "Male",
      "status": "done",
      "created_at": "2026-02-08T10:30:00Z"
    },
    {
      "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "voice_name": "Processing Voice",
      "language_tag": "English",
      "gender": "Female",
      "status": "processing",
      "created_at": "2026-02-08T10:35:00Z"
    }
  ]
}
```

### Clone voice

`POST /v1/minimax/voices/clone`

Docs: https://vibi.pro/docs/minimax-voices/clone-voice

Clone a voice from an audio file. Processing is async - the response returns immediately with status "processing". Poll the list endpoint to check when status becomes "done".

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Request body** (`multipart/form-data`, required)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | Audio file to clone voice from (max 20MB) |
| `voice_name` | string | yes | Name for the cloned voice |
| `language_tag` | string | no | Language of the voice Default: English. |
| `gender` | string | no | Gender of the voice Default: Male. |
| `need_noise_reduction` | boolean | no | Apply noise reduction to the audio Default: true. |
| `preview_text` | string | no | Text for generating a preview sample (max 200 characters) Default: Hello. |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/minimax/voices/clone" \
  -H "xi-api-key: YOUR_API_KEY" \
  -F "file=@voice_sample.mp3" \
  -F "voice_name=My Voice" \
  -F "language_tag=English" \
  -F "gender=Male" \
  -F "preview_text=Hello, this is a test"
```

**Example (JavaScript)**

```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('voice_name', 'My Voice');
formData.append('language_tag', 'English');
formData.append('gender', 'Male');
formData.append('preview_text', 'Hello, this is a test');

const response = await fetch('https://api.vibi.pro/v1/minimax/voices/clone', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' },
  body: formData
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

with open('voice_sample.mp3', 'rb') as f:
    response = requests.post(
        'https://api.vibi.pro/v1/minimax/voices/clone',
        headers={'xi-api-key': 'YOUR_API_KEY'},
        files={'file': f},
        data={
            'voice_name': 'My Voice',
            'language_tag': 'English',
            'gender': 'Male',
            'preview_text': 'Hello, this is a test',
        },
    )
data = response.json()
```

**Responses**

- `200` Voice clone initiated (processing async)

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "voice_name": "My Voice",
  "language_tag": "English",
  "gender": "Male",
  "status": "processing",
  "created_at": "2026-02-08T10:30:00Z"
}
```

### Delete cloned voice

`DELETE /v1/minimax/voices/{id}`

Docs: https://vibi.pro/docs/minimax-voices/delete-cloned-voice

Delete a cloned voice. The voice will no longer appear in the list.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Voice ID (UUID) |

**Example (curl)**

```bash
curl -X DELETE "https://api.vibi.pro/v1/minimax/voices/{voice_id}" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/minimax/voices/{voice_id}', {
  method: 'DELETE',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.delete(
    'https://api.vibi.pro/v1/minimax/voices/{voice_id}',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `200` Voice deleted

```json
{
  "success": true,
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### Retry clone

`POST /v1/minimax/voices/{voice_id}/retry`

Docs: https://vibi.pro/docs/minimax-voices/retry-clone

Retry a failed or completed voice clone. Re-uses the original audio file and settings, re-deducts credits.

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Path parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Voice clone ID (UUID) |

**Example (curl)**

```bash
curl -X POST "https://api.vibi.pro/v1/minimax/voices/{voice_id}/retry" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
const response = await fetch('https://api.vibi.pro/v1/minimax/voices/{voice_id}/retry', {
  method: 'POST',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();
```

**Example (Python)**

```python
import requests

response = requests.post(
    'https://api.vibi.pro/v1/minimax/voices/{voice_id}/retry',
    headers={'xi-api-key': 'YOUR_API_KEY'},
)
data = response.json()
```

**Responses**

- `200` Clone re-initiated (processing async)

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "voice_name": "My Voice",
  "language_tag": "English",
  "gender": "Male",
  "status": "processing",
  "created_at": "2026-02-08T10:30:00Z"
}
```

## CapCut Voices

### List system voices

`GET /v1/capcut/system-voices`

Docs: https://vibi.pro/docs/capcut-voices/list-system-voices

Browse the CapCut voice library with filters and page-based pagination. Each returned voice_id can be used directly with the Text to speech endpoint (provider 'capcut').

**Authentication:** `xi-api-key: YOUR_API_KEY` header

**Query parameters**

| Name | Type | Required | Description |
|---|---|---|---|
| `page` | integer | no | Page number (1-indexed) |
| `page_size` | integer | no | Number of voices per page |
| `search` | string | no | Case-insensitive substring match on the voice name |
| `language` | string | no | Language code filter. Values: en, vi, zh, id, es, pt, ja, th (or omit for all) |
| `gender` | string | no | Gender filter. Values: Male, Female, Gender Diversity (or omit for all) |
| `age` | string | no | Age filter. Values: Elderly, Middle-aged, Teenager, Children, Baby, No Age (or omit for all) |
| `emotion` | string | no | Emotion filter. Values: Excited, Plain, Pleasure, Shy, Proud, Admire, Angery, Sad, Disgust, Flattery, Fear, Embarrassing, Confused (or omit for all) |
| `accent` | string | no | Tone/accent filter. Values: Gentle / Kind, Cute, Steady, Freaky, Terrorism, Stupid, Atmosphere, Sexy, Mysterious, Magnetic, Deep, ASMR, High, Weak, Technology, Mistress, Refreshing/Fierce, Chinese Style, Retro (or omit for all) |

**Example (curl)**

```bash
# Basic request (first page)
curl -X GET "https://api.vibi.pro/v1/capcut/system-voices?page=1&page_size=30" \
  -H "xi-api-key: YOUR_API_KEY"

# With filters
curl -X GET "https://api.vibi.pro/v1/capcut/system-voices\
  ?page=1\
  &page_size=30\
  &language=en\
  &gender=Male\
  &age=Middle-aged\
  &emotion=Excited\
  &accent=Deep\
  &search=narrator" \
  -H "xi-api-key: YOUR_API_KEY"
```

**Example (JavaScript)**

```javascript
// Basic request (first page)
const response = await fetch('https://api.vibi.pro/v1/capcut/system-voices?page=1&page_size=30', {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const data = await response.json();

// With filters
const params = new URLSearchParams({
  page: '1',
  page_size: '30',
  language: 'en',
  gender: 'Male',
  age: 'Middle-aged',
  emotion: 'Excited',
  accent: 'Deep',
  search: 'narrator',
});
const filtered = await fetch(`https://api.vibi.pro/v1/capcut/system-voices?${params}`, {
  method: 'GET',
  headers: { 'xi-api-key': 'YOUR_API_KEY' }
});
const filteredData = await filtered.json();
```

**Example (Python)**

```python
import requests

headers = {'xi-api-key': 'YOUR_API_KEY'}

# Basic request (first page)
response = requests.get(
    'https://api.vibi.pro/v1/capcut/system-voices',
    headers=headers,
    params={'page': 1, 'page_size': 30},
)
data = response.json()

# With filters
response = requests.get(
    'https://api.vibi.pro/v1/capcut/system-voices',
    headers=headers,
    params={
        'page': 1,
        'page_size': 30,
        'language': 'en',
        'gender': 'Male',
        'age': 'Middle-aged',
        'emotion': 'Excited',
        'accent': 'Deep',
        'search': 'narrator',
    },
)
data = response.json()
```

**Responses**

- `200` OK

```json
{
  "voice_list": [
    {
      "voice_id": "7036989371643923969",
      "name": "Energetic Narrator",
      "language": "en",
      "gender": "Male",
      "platform": "sami",
      "image_url": "https://p16-capcut.../avatar.png",
      "tags": ["male", "steady", "narration"],
      "preview_url": "https://api.vibi.pro/v1/capcut/voices/7036989371643923969/preview"
    },
    {
      "voice_id": "7036989371643923970",
      "name": "Warm Storyteller",
      "language": "en",
      "gender": "Female",
      "platform": "11labs",
      "image_url": "https://p16-capcut.../avatar2.png",
      "tags": ["female", "gentle / kind", "audiobook"],
      "preview_url": "https://api.vibi.pro/v1/capcut/voices/7036989371643923970/preview"
    }
  ],
  "total": 480,
  "has_more": true
}
```
