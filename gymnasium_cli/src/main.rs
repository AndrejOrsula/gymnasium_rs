//! CLI tool for the Rust implementation of Gymnasium.
#![allow(unused)]

use clap::Parser;
use gymnasium::prelude::*;

fn main() {
    let args = Args::parse();
}

/// Arguments for the CLI
#[derive(Parser)]
#[command(author, version, about)]
struct Args {}

#[cfg(test)]
mod tests {
    use super::*;
}
