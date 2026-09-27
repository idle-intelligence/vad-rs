# vad-rs

Rust reimplementation of [Silero VAD](https://github.com/snakers4/silero-vad) v5 inference, using [candle](https://github.com/huggingface/candle) for CPU tensor ops.

Accepts 24kHz PCM audio and emits `SpeechStart` / `SpeechEnd` events with configurable thresholds and hysteresis (redemption frames).

## Usage

```rust
use vad_rs::{SileroVad, VadDetector};

let model_bytes = std::fs::read("silero-vad-v5.safetensors")?;
let vad = SileroVad::from_bytes(&model_bytes)?;
let mut detector = VadDetector::new(vad);

// Optional: tune sensitivity
detector.set_thresholds(0.5, 0.35);      // positive, negative
detector.set_redemption_frames(8);         // frames below threshold before SpeechEnd

// Feed 24kHz audio in any chunk size
let events = detector.feed_audio(&samples_24khz);
for event in events {
    println!("{event:?}"); // SpeechStart or SpeechEnd
}
```

## Model

Download the safetensors weights from HuggingFace: [idle-intelligence/silero-vad-v5-safetensors](https://huggingface.co/idle-intelligence/silero-vad-v5-safetensors)

## API

### `SileroVad`

Low-level: processes exactly 512 samples of 16kHz audio per call, returns speech probability `[0, 1]`.

### `VadDetector`

High-level wrapper:
- Resamples 24kHz → 16kHz (linear interpolation)
- Windows into 512-sample chunks
- State machine with configurable thresholds and redemption frames
- `set_thresholds(positive, negative)` — speech start/end probability thresholds (defaults: 0.5, 0.35)
- `set_redemption_frames(n)` — consecutive below-threshold frames required for `SpeechEnd` (default: 8, ~256ms)

## Tests

Most tests require the model weights (and some a WAV file). They are `#[ignore]` by default and enabled with the `local-tests` feature.

Setup:

```bash
mkdir -p test-data
# Symlink or copy the model
ln -s /path/to/silero-vad-v5.safetensors test-data/silero-vad-v5.safetensors
# Place a 24kHz 16-bit PCM WAV for E2E tests
cp /path/to/speech.wav test-data/speech-24khz.wav
```

The model is available at [idle-intelligence/silero-vad-v5-safetensors](https://huggingface.co/idle-intelligence/silero-vad-v5-safetensors).

```bash
# All tests (requires test-data/)
cargo test --features local-tests

# Default tests only (no model needed)
cargo test
```

## License

MIT. See [LICENSE](LICENSE). Reimplements [Silero VAD](https://github.com/snakers4/silero-vad), also MIT.
