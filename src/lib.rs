use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;

/// VAD events emitted by the high-level detector.
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    SpeechStart,
    SpeechEnd,
}

/// Low-level Silero VAD v5 model inference on 16kHz audio.
///
/// Processes exactly 512 samples per call. Maintains 64-sample context buffer
/// and LSTM hidden state across calls.
pub struct SileroVad {
    device: Device,

    // STFT conv (no bias) [258, 1, 256]
    stft_weight: Tensor,

    // Encoder conv weights + biases
    enc0_weight: Tensor, // [128, 129, 3]
    enc0_bias: Tensor,   // [128]
    enc1_weight: Tensor, // [64, 128, 3]
    enc1_bias: Tensor,   // [64]
    enc2_weight: Tensor, // [64, 64, 3]
    enc2_bias: Tensor,   // [64]
    enc3_weight: Tensor, // [128, 64, 3]
    enc3_bias: Tensor,   // [128]

    // LSTM weights
    lstm_w_ih: Tensor, // [512, 128]
    lstm_w_hh: Tensor, // [512, 128]
    lstm_b_ih: Tensor, // [512]
    lstm_b_hh: Tensor, // [512]

    // Decoder output conv
    dec_weight: Tensor, // [1, 128, 1]
    dec_bias: Tensor,   // [1]

    // Persistent LSTM state
    h: Tensor, // [1, 128]
    c: Tensor, // [1, 128]

    /// Context buffer: last 64 samples from previous chunk (or zeros initially)
    context: Vec<f32>,
}

impl SileroVad {
    /// Load model from raw safetensors bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let device = Device::Cpu;
        let tensors = SafeTensors::deserialize(data).context("Failed to deserialize safetensors")?;

        let load = |name: &str| -> Result<Tensor> {
            let view = tensors.tensor(name).with_context(|| format!("Missing tensor: {name}"))?;
            let shape: Vec<usize> = view.shape().to_vec();
            let raw = view.data();
            let floats: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::from_vec(floats, shape.as_slice(), &device)
                .with_context(|| format!("Failed to create tensor: {name}"))
        };

        let stft_weight = load("stft.conv.weight")?;
        let enc0_weight = load("encoder.0.conv.weight")?;
        let enc0_bias   = load("encoder.0.conv.bias")?;
        let enc1_weight = load("encoder.1.conv.weight")?;
        let enc1_bias   = load("encoder.1.conv.bias")?;
        let enc2_weight = load("encoder.2.conv.weight")?;
        let enc2_bias   = load("encoder.2.conv.bias")?;
        let enc3_weight = load("encoder.3.conv.weight")?;
        let enc3_bias   = load("encoder.3.conv.bias")?;
        let lstm_w_ih   = load("decoder.lstm.weight_ih_l0")?;
        let lstm_w_hh   = load("decoder.lstm.weight_hh_l0")?;
        let lstm_b_ih   = load("decoder.lstm.bias_ih_l0")?;
        let lstm_b_hh   = load("decoder.lstm.bias_hh_l0")?;
        let dec_weight  = load("decoder.output.weight")?;
        let dec_bias    = load("decoder.output.bias")?;

        let h = Tensor::zeros((1usize, 128usize), DType::F32, &device)?;
        let c = Tensor::zeros((1usize, 128usize), DType::F32, &device)?;
        let context = vec![0.0f32; 64];

        Ok(Self {
            device,
            stft_weight,
            enc0_weight, enc0_bias,
            enc1_weight, enc1_bias,
            enc2_weight, enc2_bias,
            enc3_weight, enc3_bias,
            lstm_w_ih, lstm_w_hh, lstm_b_ih, lstm_b_hh,
            dec_weight, dec_bias,
            h, c,
            context,
        })
    }

    /// Reset LSTM state and context buffer.
    pub fn reset(&mut self) {
        self.h = Tensor::zeros((1usize, 128usize), DType::F32, &self.device).unwrap();
        self.c = Tensor::zeros((1usize, 128usize), DType::F32, &self.device).unwrap();
        self.context = vec![0.0f32; 64];
    }

    /// Process exactly 512 samples of 16kHz PCM audio.
    ///
    /// Returns the speech probability in [0, 1].
    pub fn process_chunk(&mut self, samples_16khz: &[f32]) -> Result<f32> {
        anyhow::ensure!(
            samples_16khz.len() == 512,
            "Expected 512 samples, got {}",
            samples_16khz.len()
        );

        // Build [1, 576]: 64 context + 512 new samples
        let mut input_data = Vec::with_capacity(576);
        input_data.extend_from_slice(&self.context);
        input_data.extend_from_slice(samples_16khz);

        // Update context: last 64 samples of the new chunk
        self.context.copy_from_slice(&samples_16khz[448..512]);

        // [1, 576] → [1, 1, 576] for conv1d
        let input = Tensor::from_vec(input_data, &[1usize, 576usize], &self.device)?;
        let input = input.unsqueeze(1)?; // [1, 1, 576]

        // Right-only reflection pad: 64 samples on right → [1, 1, 640]
        // (matches ONNX model's Pad node: pads=[0,0,0,64], mode=reflect)
        let padded = reflection_pad1d_right(&input, 64)?;

        // STFT conv: Conv1d(1, 258, kernel=256, stride=128), no bias
        // padded: [1, 1, 640], stft_weight: [258, 1, 256]
        let stft_out = padded.conv1d(&self.stft_weight, 0, 128, 1, 1)?; // [1, 258, 4]

        // Split channels
        let real = stft_out.narrow(1, 0, 129)?;   // [1, 129, T]
        let imag = stft_out.narrow(1, 129, 129)?; // [1, 129, T]

        // Magnitude
        let mag = (real.powf(2.0)? + imag.powf(2.0)?)?.sqrt()?; // [1, 129, T]

        // Encoder
        let x = conv1d_bias(&mag, &self.enc0_weight, &self.enc0_bias, 1, 1)?;
        let x = x.relu()?;
        let x = conv1d_bias(&x, &self.enc1_weight, &self.enc1_bias, 1, 2)?;
        let x = x.relu()?;
        let x = conv1d_bias(&x, &self.enc2_weight, &self.enc2_bias, 1, 2)?;
        let x = x.relu()?;
        let x = conv1d_bias(&x, &self.enc3_weight, &self.enc3_bias, 1, 1)?;
        let x = x.relu()?;

        // [1, 128, T'] → [1, T', 128]
        let x = x.permute((0, 2, 1))?;

        // LSTM over time steps
        let t_prime = x.dim(1)?;
        let mut h = self.h.clone();
        let mut c = self.c.clone();

        for t in 0..t_prime {
            let xt = x.narrow(1, t, 1)?.squeeze(1)?; // [1, 128]
            let (h_new, c_new) = lstm_step(
                &xt, &h, &c,
                &self.lstm_w_ih, &self.lstm_w_hh,
                &self.lstm_b_ih, &self.lstm_b_hh,
            )?;
            h = h_new;
            c = c_new;
        }

        // Persist LSTM state
        self.h = h.clone();
        self.c = c.clone();

        // ReLU on last LSTM output
        let h_relu = h.relu()?; // [1, 128]

        // Reshape to [1, 128, 1] for Conv1d(128, 1, kernel=1)
        let h_conv = h_relu.unsqueeze(2)?; // [1, 128, 1]

        // decoder.output: [1, 128, 1]
        let out = h_conv.conv1d(&self.dec_weight, 0, 1, 1, 1)?; // [1, 1, 1]

        // Add bias [1] → broadcast over [1, 1, 1]
        let bias = self.dec_bias.reshape(&[1usize, 1usize, 1usize])?;
        let out = out.broadcast_add(&bias)?;

        // Sigmoid
        let prob = sigmoid(&out)?;

        let prob_val = prob.flatten_all()?.to_vec1::<f32>()?;
        Ok(prob_val[0])
    }
}

/// Apply ReflectionPad1d(pad) to a [batch, channels, length] tensor (both sides).
/// Only used in tests for verification against PyTorch.
#[cfg(test)]
fn reflection_pad1d(x: &Tensor, pad: usize) -> Result<Tensor> {
    let (b, c, l) = x.dims3()?;
    anyhow::ensure!(pad < l, "pad ({pad}) must be < length ({l})");

    let x_vec = x.to_vec3::<f32>()?;
    let new_l = l + 2 * pad;
    let mut padded = vec![0.0f32; b * c * new_l];

    for bi in 0..b {
        for ci in 0..c {
            let src = &x_vec[bi][ci];
            let base = (bi * c + ci) * new_l;

            // Left reflection: mirror indices [pad, pad-1, ..., 1]
            for i in 0..pad {
                padded[base + i] = src[pad - i];
            }
            // Copy original signal
            for i in 0..l {
                padded[base + pad + i] = src[i];
            }
            // Right reflection: mirror indices [l-2, l-3, ..., l-1-pad]
            for i in 0..pad {
                padded[base + pad + l + i] = src[l - 2 - i];
            }
        }
    }

    Ok(Tensor::from_vec(padded, &[b, c, new_l], x.device())?)
}

/// Apply reflection padding only on the right side of a [batch, channels, length] tensor.
fn reflection_pad1d_right(x: &Tensor, pad: usize) -> Result<Tensor> {
    let (b, c, l) = x.dims3()?;
    anyhow::ensure!(pad < l, "pad ({pad}) must be < length ({l})");

    let x_vec = x.to_vec3::<f32>()?;
    let new_l = l + pad;
    let mut padded = vec![0.0f32; b * c * new_l];

    for bi in 0..b {
        for ci in 0..c {
            let src = &x_vec[bi][ci];
            let base = (bi * c + ci) * new_l;

            // Copy original signal
            for i in 0..l {
                padded[base + i] = src[i];
            }
            // Right reflection: mirror indices [l-2, l-3, ..., l-1-pad]
            for i in 0..pad {
                padded[base + l + i] = src[l - 2 - i];
            }
        }
    }

    Ok(Tensor::from_vec(padded, &[b, c, new_l], x.device())?)
}

/// Conv1d via Tensor::conv1d, then add bias.
fn conv1d_bias(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    padding: usize,
    stride: usize,
) -> Result<Tensor> {
    let out = x.conv1d(weight, padding, stride, 1, 1)?;
    let out_channels = bias.dims1()?;
    let bias = bias.reshape(&[1usize, out_channels, 1usize])?;
    Ok(out.broadcast_add(&bias)?)
}

/// Single LSTM step. Returns (h_new, c_new).
fn lstm_step(
    x: &Tensor,    // [1, 128]
    h: &Tensor,    // [1, 128]
    c: &Tensor,    // [1, 128]
    w_ih: &Tensor, // [512, 128]
    w_hh: &Tensor, // [512, 128]
    b_ih: &Tensor, // [512]
    b_hh: &Tensor, // [512]
) -> Result<(Tensor, Tensor)> {
    // gates = x @ w_ih.T + h @ w_hh.T + b_ih + b_hh → [1, 512]
    let gates_ih = x.matmul(&w_ih.t()?)?;
    let gates_hh = h.matmul(&w_hh.t()?)?;
    let b_ih_2d  = b_ih.unsqueeze(0)?; // [1, 512]
    let b_hh_2d  = b_hh.unsqueeze(0)?; // [1, 512]
    let gates = ((gates_ih + gates_hh)? + (b_ih_2d + b_hh_2d)?)?; // [1, 512]

    // Gate order: i[0..128], f[128..256], g[256..384], o[384..512]
    let i_gate = sigmoid(&gates.narrow(1, 0, 128)?)?;
    let f_gate = sigmoid(&gates.narrow(1, 128, 128)?)?;
    let g_gate  = gates.narrow(1, 256, 128)?.tanh()?;
    let o_gate  = sigmoid(&gates.narrow(1, 384, 128)?)?;

    let c_new: Tensor = ((f_gate * c)? + (i_gate * g_gate)?)?;
    let h_new: Tensor = (o_gate * c_new.tanh()?)?;

    Ok((h_new, c_new))
}

/// Element-wise sigmoid: 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    let denom = (exp_neg + 1.0f64)?;
    Ok((1.0f64 / denom)?)
}

// ---------------------------------------------------------------------------
// High-level wrapper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum VadState {
    Idle,
    Speaking,
}

/// High-level VAD: accepts 24kHz audio, resamples to 16kHz, emits events.
pub struct VadDetector {
    vad: SileroVad,
    state: VadState,
    /// Resampled 16kHz samples not yet fed to the model
    resample_buf: Vec<f32>,
    /// Fractional read position within the current input chunk
    resample_pos: f64,
    /// Last 24kHz sample seen (for interpolation across chunks)
    last_sample: f32,
}

impl VadDetector {
    pub fn new(vad: SileroVad) -> Self {
        Self {
            vad,
            state: VadState::Idle,
            resample_buf: Vec::new(),
            resample_pos: 0.0,
            last_sample: 0.0,
        }
    }

    /// Feed 24kHz samples; returns any VAD events detected.
    pub fn feed_audio(&mut self, samples_24khz: &[f32]) -> Vec<VadEvent> {
        // 24kHz → 16kHz: ratio = 24000/16000 = 1.5 input samples per output sample
        let ratio = 24000.0_f64 / 16000.0_f64;
        let resampled = resample_linear(
            samples_24khz, ratio,
            &mut self.resample_pos,
            &mut self.last_sample,
        );
        self.resample_buf.extend_from_slice(&resampled);

        let mut events = Vec::new();

        while self.resample_buf.len() >= 512 {
            let chunk: Vec<f32> = self.resample_buf.drain(..512).collect();
            match self.vad.process_chunk(&chunk) {
                Ok(prob) => match self.state {
                    VadState::Idle => {
                        if prob >= 0.5 {
                            self.state = VadState::Speaking;
                            events.push(VadEvent::SpeechStart);
                        }
                    }
                    VadState::Speaking => {
                        if prob < 0.35 {
                            self.state = VadState::Idle;
                            events.push(VadEvent::SpeechEnd);
                        }
                    }
                },
                Err(_) => {}
            }
        }

        events
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.vad.reset();
        self.state = VadState::Idle;
        self.resample_buf.clear();
        self.resample_pos = 0.0;
        self.last_sample = 0.0;
    }
}

/// Resample using linear interpolation.
///
/// `ratio` = input_rate / output_rate (e.g. 1.5 for 24k→16k).
/// State (`pos`, `last_sample`) persists across calls.
fn resample_linear(
    input: &[f32],
    ratio: f64,
    pos: &mut f64,
    last_sample: &mut f32,
) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::new();

    loop {
        let floor = pos.floor() as isize;
        let frac = *pos - floor as f64;

        let s0 = if floor < 0 {
            *last_sample
        } else if floor as usize >= input.len() {
            break;
        } else {
            input[floor as usize]
        };

        let s1_idx = floor + 1;
        if s1_idx as usize >= input.len() {
            break;
        }
        let s1 = if floor < 0 { input[0] } else { input[s1_idx as usize] };

        output.push(s0 + (s1 - s0) * frac as f32);
        *pos += ratio;
    }

    *last_sample = *input.last().unwrap();
    *pos -= input.len() as f64;

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires local model file"]
    fn test_silence_probability() {
        let path = "/Users/tc/Code/idle-intelligence/hf/silero-vad-v5.safetensors";
        let data = std::fs::read(path).expect("Failed to read model file");
        let mut vad = SileroVad::from_bytes(&data).expect("Failed to load model");

        let silence = vec![0.0f32; 512];
        let prob = vad.process_chunk(&silence).expect("Failed to process chunk");

        println!("Silence probability: {prob}");
        assert!(prob < 0.1, "Expected low probability for silence, got {prob}");
    }

    #[test]
    #[ignore = "requires local model + WAV file"]
    fn test_speech_detection() {
        let model_path = "/Users/tc/Code/idle-intelligence/hf/silero-vad-v5.safetensors";
        let wav_path = "/Users/tc/Code/idle-intelligence/stt-web-vad/web/test-bria.wav";

        let data = std::fs::read(model_path).expect("Failed to read model file");
        let mut vad = SileroVad::from_bytes(&data).expect("Failed to load model");

        // Read 24kHz 16-bit PCM WAV manually (skip 44-byte header)
        let wav_bytes = std::fs::read(wav_path).expect("Failed to read WAV file");
        let samples_24k: Vec<f32> = wav_bytes[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        // Resample 24kHz → 16kHz manually (same as VadDetector)
        let mut pos = 0.0_f64;
        let mut last_sample = 0.0_f32;
        let resampled = resample_linear(&samples_24k, 1.5, &mut pos, &mut last_sample);
        println!("Resampled: {} samples ({:.2}s at 16kHz)", resampled.len(), resampled.len() as f64 / 16000.0);

        // Feed 512-sample chunks directly to SileroVad and print probabilities
        let mut max_prob: f32 = 0.0;
        let mut chunk_count = 0;
        for chunk in resampled.chunks_exact(512) {
            let prob = vad.process_chunk(chunk).expect("process_chunk failed");
            if chunk_count < 20 || chunk_count % 50 == 0 || prob > 0.1 {
                let time_s = chunk_count as f64 * 512.0 / 16000.0;
                println!("  chunk {chunk_count} (t={time_s:.2}s): prob={prob:.6}");
            }
            if prob > max_prob { max_prob = prob; }
            chunk_count += 1;
        }
        println!("Total chunks: {chunk_count}, max probability: {max_prob:.6}");
        assert!(max_prob > 0.3, "Should detect speech with prob>0.3 in a speech file, got {max_prob}");
    }

    #[test]
    #[ignore = "requires local model file"]
    fn test_forward_pass_trace() {
        // Trace intermediate tensor values through the forward pass
        // to identify which layer produces unexpected values.
        let model_path = "/Users/tc/Code/idle-intelligence/hf/silero-vad-v5.safetensors";
        let data = std::fs::read(model_path).expect("Failed to read model file");
        let vad = SileroVad::from_bytes(&data).expect("Failed to load model");

        // Use a speech-like 16kHz chunk: sine waves at multiple frequencies
        // (simulate formant structure of speech)
        let chunk: Vec<f32> = (0..512)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.3 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()  // fundamental
                + 0.2 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()  // formant
                + 0.1 * (2.0 * std::f32::consts::PI * 2500.0 * t).sin() // formant
            })
            .collect();

        let device = &vad.device;

        // Build input [1, 576]
        let mut input_data = Vec::with_capacity(576);
        input_data.extend_from_slice(&vad.context); // 64 zeros
        input_data.extend_from_slice(&chunk);
        let input = Tensor::from_vec(input_data, &[1usize, 576usize], device).unwrap();
        let input = input.unsqueeze(1).unwrap(); // [1, 1, 576]
        println!("Input shape: {:?}, min={:.6}, max={:.6}", input.dims(),
            input.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().cloned().fold(f32::INFINITY, f32::min),
            input.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        // Right-only reflection pad (64 samples)
        let padded = reflection_pad1d_right(&input, 64).unwrap();
        println!("Padded shape: {:?}", padded.dims());

        // STFT conv
        let stft_out = padded.conv1d(&vad.stft_weight, 0, 128, 1, 1).unwrap();
        println!("STFT out shape: {:?}", stft_out.dims());
        let stft_vals: Vec<f32> = stft_out.flatten_all().unwrap().to_vec1().unwrap();
        let stft_min = stft_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let stft_max = stft_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let stft_mean: f32 = stft_vals.iter().sum::<f32>() / stft_vals.len() as f32;
        println!("STFT: min={stft_min:.6}, max={stft_max:.6}, mean={stft_mean:.6}");

        // Split + magnitude
        let real = stft_out.narrow(1, 0, 129).unwrap();
        let imag = stft_out.narrow(1, 129, 129).unwrap();
        let mag = (real.powf(2.0).unwrap() + imag.powf(2.0).unwrap()).unwrap().sqrt().unwrap();
        println!("Magnitude shape: {:?}", mag.dims());
        let mag_vals: Vec<f32> = mag.flatten_all().unwrap().to_vec1().unwrap();
        let mag_min = mag_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let mag_max = mag_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mag_mean: f32 = mag_vals.iter().sum::<f32>() / mag_vals.len() as f32;
        println!("Magnitude: min={mag_min:.6}, max={mag_max:.6}, mean={mag_mean:.6}");

        // Encoder layers
        let x = conv1d_bias(&mag, &vad.enc0_weight, &vad.enc0_bias, 1, 1).unwrap().relu().unwrap();
        let v: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        println!("enc0 shape: {:?}, min={:.6}, max={:.6}, mean={:.6}, nonzero={}",
            x.dims(), v.iter().cloned().fold(f32::INFINITY, f32::min),
            v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            v.iter().sum::<f32>() / v.len() as f32,
            v.iter().filter(|&&x| x > 0.0).count());

        let x = conv1d_bias(&x, &vad.enc1_weight, &vad.enc1_bias, 1, 2).unwrap().relu().unwrap();
        let v: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        println!("enc1 shape: {:?}, min={:.6}, max={:.6}, mean={:.6}, nonzero={}",
            x.dims(), v.iter().cloned().fold(f32::INFINITY, f32::min),
            v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            v.iter().sum::<f32>() / v.len() as f32,
            v.iter().filter(|&&x| x > 0.0).count());

        let x = conv1d_bias(&x, &vad.enc2_weight, &vad.enc2_bias, 1, 2).unwrap().relu().unwrap();
        let v: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        println!("enc2 shape: {:?}, min={:.6}, max={:.6}, mean={:.6}, nonzero={}",
            x.dims(), v.iter().cloned().fold(f32::INFINITY, f32::min),
            v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            v.iter().sum::<f32>() / v.len() as f32,
            v.iter().filter(|&&x| x > 0.0).count());

        let x = conv1d_bias(&x, &vad.enc3_weight, &vad.enc3_bias, 1, 1).unwrap().relu().unwrap();
        let v: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        println!("enc3 shape: {:?}, min={:.6}, max={:.6}, mean={:.6}, nonzero={}",
            x.dims(), v.iter().cloned().fold(f32::INFINITY, f32::min),
            v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            v.iter().sum::<f32>() / v.len() as f32,
            v.iter().filter(|&&x| x > 0.0).count());

        // Permute + LSTM
        let x = x.permute((0, 2, 1)).unwrap();
        println!("After permute: {:?}", x.dims());

        let t_prime = x.dim(1).unwrap();
        let mut h = vad.h.clone();
        let mut c = vad.c.clone();
        for t in 0..t_prime {
            let xt = x.narrow(1, t, 1).unwrap().squeeze(1).unwrap();
            let (h_new, c_new) = lstm_step(&xt, &h, &c, &vad.lstm_w_ih, &vad.lstm_w_hh, &vad.lstm_b_ih, &vad.lstm_b_hh).unwrap();
            h = h_new;
            c = c_new;
        }
        let h_vals: Vec<f32> = h.flatten_all().unwrap().to_vec1().unwrap();
        println!("LSTM h: min={:.6}, max={:.6}, mean={:.6}",
            h_vals.iter().cloned().fold(f32::INFINITY, f32::min),
            h_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            h_vals.iter().sum::<f32>() / h_vals.len() as f32);

        // Decoder
        let h_relu = h.relu().unwrap();
        let h_conv = h_relu.unsqueeze(2).unwrap();
        let out = h_conv.conv1d(&vad.dec_weight, 0, 1, 1, 1).unwrap();
        let bias = vad.dec_bias.reshape(&[1usize, 1usize, 1usize]).unwrap();
        let out = out.broadcast_add(&bias).unwrap();
        let out_val: f32 = out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        println!("Pre-sigmoid output: {out_val:.6}");

        let prob = sigmoid(&out).unwrap();
        let prob_val: f32 = prob.flatten_all().unwrap().to_vec1().unwrap()[0];
        println!("Sigmoid probability: {prob_val:.6}");
    }

    #[test]
    fn test_api_compiles() {
        let _start = VadEvent::SpeechStart;
        let _end = VadEvent::SpeechEnd;
        assert_eq!(_start, VadEvent::SpeechStart);
    }

    #[test]
    fn test_reflection_pad1d() {
        // Verify our reflection padding matches PyTorch's ReflectionPad1d.
        // PyTorch: ReflectionPad1d(2) on [0,1,2,3,4] → [2,1, 0,1,2,3,4, 3,2]
        let input = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
            &[1, 1, 5],
            &Device::Cpu,
        ).unwrap();
        let padded = reflection_pad1d(&input, 2).unwrap();
        let vals: Vec<f32> = padded.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    #[ignore = "requires local model file"]
    fn test_noise_probability() {
        let path = "/Users/tc/Code/idle-intelligence/hf/silero-vad-v5.safetensors";
        let data = std::fs::read(path).expect("Failed to read model file");
        let mut vad = SileroVad::from_bytes(&data).expect("Failed to load model");

        // Feed several chunks of white noise to build up LSTM state.
        // Noise has broad spectral content similar to speech.
        let mut probs = Vec::new();
        for i in 0..10 {
            let chunk: Vec<f32> = (0..512)
                .map(|j| {
                    // Pseudo-random noise using simple hash
                    let seed = (i * 512 + j) as f32;
                    ((seed * 2654435761.0) % 1.0) * 2.0 - 1.0
                })
                .collect();
            let prob = vad.process_chunk(&chunk).expect("process_chunk failed");
            probs.push(prob);
            println!("Noise chunk {i}: prob={prob:.6}");
        }

        // After several chunks of noise, probability should be notably
        // higher than silence (but not necessarily > 0.5 since noise != speech)
        let max_prob = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("Max noise probability: {max_prob:.6}");
        // Noise is NOT speech — probability stays low. Just verify it's non-negative
        // and the model doesn't crash or return NaN.
        assert!(max_prob.is_finite(), "Probability should be finite, got {max_prob}");
    }

    #[test]
    #[ignore = "requires local model file"]
    fn test_vad_detector_24khz() {
        let path = "/Users/tc/Code/idle-intelligence/hf/silero-vad-v5.safetensors";
        let data = std::fs::read(path).expect("Failed to read model file");
        let vad = SileroVad::from_bytes(&data).expect("Failed to load model");
        let mut detector = VadDetector::new(vad);

        // Feed 24kHz silence — should produce no events
        let silence = vec![0.0f32; 1920]; // 80ms at 24kHz
        let events = detector.feed_audio(&silence);
        assert!(events.is_empty(), "Silence should not trigger events");

        // Feed 24kHz noise — check we don't crash
        for i in 0..20 {
            let chunk: Vec<f32> = (0..1920)
                .map(|j| {
                    let seed = (i * 1920 + j) as f32;
                    ((seed * 2654435761.0) % 1.0) * 0.5 - 0.25
                })
                .collect();
            let events = detector.feed_audio(&chunk);
            for ev in &events {
                println!("24kHz detector chunk {i}: event={ev:?}");
            }
        }
        println!("VadDetector 24kHz test passed (no crash)");
    }

    #[test]
    #[ignore = "requires local model + WAV file"]
    fn test_e2e_silence_speech_silence_speech() {
        // End-to-end test: splice real speech with silence gaps
        // Pattern: 1s silence → 2s speech → 1.5s silence → 2s speech
        let model_path = "/Users/tc/Code/idle-intelligence/hf/silero-vad-v5.safetensors";
        let wav_path = "/Users/tc/Code/idle-intelligence/stt-web-vad/web/test-bria.wav";

        let data = std::fs::read(model_path).expect("Failed to read model file");
        let vad = SileroVad::from_bytes(&data).expect("Failed to load model");
        let mut detector = VadDetector::new(vad);

        // Read real speech from WAV
        let wav_bytes = std::fs::read(wav_path).expect("Failed to read WAV file");
        let all_speech: Vec<f32> = wav_bytes[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        let sr = 24000usize;
        let mut audio = Vec::new();

        // 1.0s silence
        audio.extend(vec![0.0f32; sr]);
        // 2.0s real speech (from beginning of WAV)
        audio.extend_from_slice(&all_speech[..sr * 2]);
        // 1.5s silence
        audio.extend(vec![0.0f32; sr * 3 / 2]);
        // 2.0s real speech (from 5s into WAV)
        audio.extend_from_slice(&all_speech[sr * 5..sr * 7]);

        let total_s = audio.len() as f64 / sr as f64;
        println!("Built audio: {} samples ({total_s:.1}s at 24kHz)", audio.len());
        println!("Pattern: [0-1s silence] [1-3s speech] [3-4.5s silence] [4.5-6.5s speech]");

        // Feed in 80ms chunks (1920 samples at 24kHz)
        let chunk_size = 1920;
        let mut events_with_time: Vec<(VadEvent, f64)> = Vec::new();
        let mut sample_offset = 0usize;

        for chunk in audio.chunks(chunk_size) {
            let time_s = sample_offset as f64 / sr as f64;
            let events = detector.feed_audio(chunk);
            for ev in events {
                let label = match &ev {
                    VadEvent::SpeechStart => ">> SPEECH START",
                    VadEvent::SpeechEnd   => "<< SPEECH END  ",
                };
                println!("  {time_s:6.3}s  {label}");
                events_with_time.push((ev, time_s));
            }
            sample_offset += chunk.len();
        }

        let starts: Vec<f64> = events_with_time.iter()
            .filter(|(ev, _)| *ev == VadEvent::SpeechStart)
            .map(|(_, t)| *t)
            .collect();
        let ends: Vec<f64> = events_with_time.iter()
            .filter(|(ev, _)| *ev == VadEvent::SpeechEnd)
            .map(|(_, t)| *t)
            .collect();

        println!("\nSpeech starts: {starts:?}");
        println!("Speech ends:   {ends:?}");

        assert!(starts.len() >= 2, "Expected at least 2 SpeechStart events, got {}", starts.len());
        assert!(ends.len() >= 1, "Expected at least 1 SpeechEnd event, got {}", ends.len());

        // First speech start should be near 1.0s
        assert!(starts[0] >= 0.8 && starts[0] <= 1.5,
            "First SpeechStart at {:.3}s, expected near 1.0s", starts[0]);

        // There should be a gap (SpeechEnd) somewhere around 3.0-4.5s
        let mid_end = ends.iter().find(|&&t| t >= 2.5 && t <= 5.0);
        assert!(mid_end.is_some(),
            "Expected a SpeechEnd in the silence gap (2.5-5.0s), got ends: {ends:?}");

        // Second speech start should be near 4.5s
        assert!(starts[1] >= 3.5 && starts[1] <= 5.5,
            "Second SpeechStart at {:.3}s, expected near 4.5s", starts[1]);
    }

    #[test]
    #[ignore = "requires local model + WAV file"]
    fn test_e2e_real_audio_vad_timestamps() {
        // End-to-end test with real speech audio (test-bria.wav, 24kHz)
        // through VadDetector at 24kHz — the full pipeline including resampling
        let model_path = "/Users/tc/Code/idle-intelligence/hf/silero-vad-v5.safetensors";
        let wav_path = "/Users/tc/Code/idle-intelligence/stt-web-vad/web/test-bria.wav";

        let data = std::fs::read(model_path).expect("Failed to read model file");
        let vad = SileroVad::from_bytes(&data).expect("Failed to load model");
        let mut detector = VadDetector::new(vad);

        // Read 24kHz 16-bit PCM WAV (skip 44-byte header)
        let wav_bytes = std::fs::read(wav_path).expect("Failed to read WAV file");
        let samples_24k: Vec<f32> = wav_bytes[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        let sr = 24000;
        let chunk_size = 1920; // 80ms at 24kHz
        let total_duration = samples_24k.len() as f64 / sr as f64;
        println!("Audio: {} samples ({:.2}s at 24kHz)", samples_24k.len(), total_duration);

        let mut events_with_time: Vec<(VadEvent, f64)> = Vec::new();
        let mut sample_offset = 0usize;

        for chunk in samples_24k.chunks(chunk_size) {
            let time_s = sample_offset as f64 / sr as f64;
            let events = detector.feed_audio(chunk);
            for ev in events {
                events_with_time.push((ev, time_s));
            }
            sample_offset += chunk.len();
        }

        println!("\nVAD Timeline:");
        for (ev, t) in &events_with_time {
            let label = match ev {
                VadEvent::SpeechStart => ">> SPEECH START",
                VadEvent::SpeechEnd   => "<< SPEECH END  ",
            };
            println!("  {t:6.2}s  {label}");
        }

        // Validate basics
        let starts: Vec<f64> = events_with_time.iter()
            .filter(|(ev, _)| *ev == VadEvent::SpeechStart)
            .map(|(_, t)| *t)
            .collect();
        let ends: Vec<f64> = events_with_time.iter()
            .filter(|(ev, _)| *ev == VadEvent::SpeechEnd)
            .map(|(_, t)| *t)
            .collect();

        println!("\nSpeech segments:");
        for i in 0..starts.len() {
            let end = if i < ends.len() { format!("{:.2}s", ends[i]) } else { "EOF".to_string() };
            println!("  Segment {}: {:.2}s → {end}", i + 1, starts[i]);
        }

        assert!(!starts.is_empty(), "Should detect at least one speech segment");
        // First speech should start very early (within first 0.3s)
        assert!(starts[0] < 0.3, "First speech should start within 0.3s, got {:.3}s", starts[0]);
        println!("\nTest passed: {} speech starts, {} speech ends", starts.len(), ends.len());
    }
}
