use std::{
    io::{Read, Write},
    path::PathBuf,
    process::exit,
};

use anyhow::Result;
use clap::{Parser, ValueHint};
use df::tract::*;
use ndarray::prelude::*;

#[cfg(all(
    not(windows),
    not(target_os = "android"),
    not(target_os = "macos"),
    not(target_os = "freebsd"),
    not(target_env = "musl"),
    not(target_arch = "riscv64"),
    feature = "use-jemalloc"
))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Simple program to pipe raw audio through DeepFilterNet
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    /// Enable post-filter
    #[arg(long = "pf")]
    post_filter: bool,
    /// Post-filter beta. Higher beta results in stronger attenuation.
    #[arg(long = "pf-beta", default_value_t = 0.02)]
    post_filter_beta: f32,
    /// Compensate delay of STFT and model lookahead
    #[arg(short = 'D', long)]
    compensate_delay: bool,
    /// Attenuation limit in dB by mixing the enhanced signal with the noisy signal.
    /// An attenuation limit of 0 dB means no noise reduction will be performed, 100 dB means full
    /// noise reduction, i.e. no attenuation limit.
    #[arg(short, long, default_value_t = 100.)]
    atten_lim_db: f32,
    /// Min dB local SNR threshold for running the decoder DNN side
    #[arg(long, value_parser, allow_negative_numbers = true, default_value_t = -15.)]
    min_db_thresh: f32,
    /// Max dB local SNR threshold for running ERB decoder
    #[arg(
        long,
        value_parser,
        allow_negative_numbers = true,
        default_value_t = 35.
    )]
    max_db_erb_thresh: f32,
    /// Max dB local SNR threshold for running DF decoder
    #[arg(
        long,
        value_parser,
        allow_negative_numbers = true,
        default_value_t = 35.
    )]
    max_db_df_thresh: f32,
    /// If used with multiple channels, reduce the mask with max (1) or mean (2)
    #[arg(long, value_parser, default_value_t = 1)]
    reduce_mask: i32,
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,

    /// Number of channels
    #[arg(long, default_value_t = 1)]
    channels: usize,
    /// Sample rate (must match model)
    #[arg(long, default_value_t = 48000)]
    sr: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(level)
        .filter_module("tract_onnx", tract_level)
        .filter_module("tract_hir", tract_level)
        .filter_module("tract_core", tract_level)
        .filter_module("tract_linalg", tract_level)
        .init();

    let mut r_params = RuntimeParams::default();
    r_params = r_params.with_atten_lim(args.atten_lim_db).with_thresholds(
        args.min_db_thresh,
        args.max_db_erb_thresh,
        args.max_db_df_thresh,
    );
    if args.post_filter {
        r_params = r_params.with_post_filter(args.post_filter_beta);
    }
    if let Ok(red) = args.reduce_mask.try_into() {
        r_params = r_params.with_mask_reduce(red);
    } else {
        log::warn!("Input not valid for `reduce_mask`.")
    }
    r_params.n_ch = args.channels;

    let df_params = if let Some(tar) = args.model.as_ref() {
        match DfParams::new(tar.clone()) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Error opening model {}: {}", tar.display(), e);
                exit(1)
            }
        }
    } else if cfg!(any(feature = "default-model", feature = "default-model-ll")) {
        DfParams::default()
    } else {
        log::error!("deep-filter was not compiled with a default model. Please provide a model via '--model <path-to-model.tar.gz>'");
        exit(2)
    };

    let mut model = DfTract::new(df_params.clone(), &r_params)?;
    let sr = model.sr;
    if args.sr != sr {
        log::error!(
            "Input sample rate {} does not match model sample rate {}.",
            args.sr,
            sr
        );
        exit(1);
    }

    let hop_size = model.hop_size;
    let n_channels = args.channels;
    let mut delay = model.fft_size - model.hop_size; // STFT delay
    delay += model.lookahead * model.hop_size; // Add model latency due to lookahead
    let mut samples_to_drop = if args.compensate_delay { delay } else { 0 };

    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();

    let mut in_frame = Array2::<f32>::zeros((n_channels, hop_size));
    let mut out_frame = Array2::<f32>::zeros((n_channels, hop_size));

    let bytes_per_sample = 4; // f32le
    let frame_len_samples = n_channels * hop_size;
    let frame_len_bytes = frame_len_samples * bytes_per_sample;
    let mut read_buffer = vec![0u8; frame_len_bytes];
    let mut write_buffer = vec![0u8; frame_len_bytes];

    loop {
        let mut read_offset = 0;
        while read_offset < frame_len_bytes {
            match stdin.read(&mut read_buffer[read_offset..]) {
                Ok(0) => break,
                Ok(n) => read_offset += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        if read_offset == 0 {
            break;
        }
        if read_offset < frame_len_bytes {
            read_buffer[read_offset..].fill(0);
        }

        let mut sample_idx = 0;
        for chunk in read_buffer.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            let ch = sample_idx % n_channels;
            let t = sample_idx / n_channels;
            in_frame[[ch, t]] = val;
            sample_idx += 1;
        }

        model.process(in_frame.view(), out_frame.view_mut())?;

        let mut start_t = 0;
        if samples_to_drop > 0 {
            if samples_to_drop >= hop_size {
                samples_to_drop -= hop_size;
                continue;
            } else {
                start_t = samples_to_drop;
                samples_to_drop = 0;
            }
        }

        let mut write_offset = 0;
        for t in start_t..hop_size {
            for ch in 0..n_channels {
                let val = out_frame[[ch, t]];
                let bytes = val.to_le_bytes();
                write_buffer[write_offset..write_offset + 4].copy_from_slice(&bytes);
                write_offset += 4;
            }
        }
        stdout.write_all(&write_buffer[..write_offset])?;
    }
    stdout.flush()?;

    Ok(())
}
