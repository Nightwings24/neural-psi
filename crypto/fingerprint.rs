//! Run the REAL flash-psi (FLPSI) protocol on REAL fingerprint codes.
//!
//! Reads a plaintext file describing Bob's database and Alice's query, runs the
//! actual masked-OPRF + garbled-circuit + VOLE + Shamir protocol, and prints
//! which database entries Alice matched (genuine accept / impostor reject).
//!
//! Input file format (all vectors are 128 chars of '0'/'1'):
//!   line 0:  "<m> <d>"          m = db size, d must be 128
//!   line 1:  query bits
//!   line 2.. db bits  (m lines)
//!
//! Usage:
//!   fingerprint <input_file> [-w WEIGHT] [-s SS_COUNT] [-t T]

use clap::Parser;
use flash_psi::*;
use ocelot::svole::wykw;
use scuttlebutt::{
    field::{F128b, F64b},
    AesRng, Channel,
};
use std::io::{BufRead, BufReader, BufWriter};
use std::time::Instant;

#[cfg(not(target_os = "windows"))]
use std::os::unix::net::UnixStream;

#[derive(Parser, Debug)]
#[clap(about = "real FLPSI on real fingerprint codes")]
struct Cli {
    input: String,
    #[clap(short, long, default_value_t = 14)]
    weight: usize,
    #[clap(short, long, default_value_t = 64)]
    subsample_count: usize,
    #[clap(short, default_value_t = 2)]
    t: usize,
    #[clap(short, long, default_value_t = false)]
    verbose: bool,
}

fn parse_bits(line: &str) -> Vec<u8> {
    line.trim()
        .chars()
        .map(|c| if c == '1' { 1u8 } else { 0u8 })
        .collect()
}

fn main() {
    let cli = Cli::parse();
    let f = std::fs::File::open(&cli.input).expect("cannot open input file");
    let mut lines = BufReader::new(f).lines().map(|l| l.unwrap());

    let header = lines.next().expect("missing header");
    let mut it = header.split_whitespace();
    let m: usize = it.next().unwrap().parse().unwrap();
    let d: usize = it.next().unwrap().parse().unwrap();
    assert_eq!(d, 128, "protocol requires 128-bit codes");

    let query: Vec<bool> = parse_bits(&lines.next().expect("missing query"))
        .into_iter()
        .map(|x| x == 1)
        .collect();
    assert_eq!(query.len(), 128, "query must be 128 bits");

    let db: Vec<Vec<u8>> = (0..m)
        .map(|_| parse_bits(&lines.next().expect("missing db row")))
        .collect();
    for (i, row) in db.iter().enumerate() {
        assert_eq!(row.len(), 128, "db row {i} must be 128 bits");
    }

    // ---- run the REAL FLPSI protocol ----
    let mut rng = AesRng::new();
    let (ot_sender, ot_receiver) =
        ot::setup_dummy_voleot::<wykw::Sender<F128b>, wykw::Receiver<F128b>, _>(&mut rng);

    let (mut fl_sender, mut fl_receiver, _orig) =
        flpsi::setup_flpsi_with_inputs::<F64b, F64b, _, _, _, false>(
            cli.weight,
            cli.subsample_count,
            cli.t,
            db,
            ot_sender,
            ot_receiver,
            &mut rng,
            cli.verbose,
        );

    let upto = m - 1; // search the whole database
    let (sender, receiver) = UnixStream::pair().unwrap();

    let start = Instant::now();
    let handle = std::thread::spawn(move || {
        let reader = BufReader::new(sender.try_clone().unwrap());
        let writer = BufWriter::new(sender);
        let mut chan = Channel::new(reader, writer);
        let mut rng = AesRng::new();
        fl_receiver
            .receive(upto, &query, &mut chan, &mut rng, cli.verbose)
            .unwrap()
    });

    let reader = BufReader::new(receiver.try_clone().unwrap());
    let writer = BufWriter::new(receiver);
    let mut chan = Channel::new(reader, writer);
    fl_sender.send(&mut chan, &mut rng).unwrap();
    let matches = handle.join().unwrap();
    let elapsed = start.elapsed();

    // ---- report ----
    let matched_idx: Vec<usize> = matches.iter().map(|(i, _)| *i).collect();
    println!("REAL_FLPSI m={m} weight={} ss={} t={}", cli.weight, cli.subsample_count, cli.t);
    println!("MATCHED_INDICES {matched_idx:?}");
    println!("NUM_MATCHES {}", matched_idx.len());
    println!("DURATION_MS {:.3}", elapsed.as_secs_f64() * 1000.0);
}
