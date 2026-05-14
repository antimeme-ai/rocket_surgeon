use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser)]
#[command(name = "xtask", about = "rocket-surgeon build tasks")]
enum Xtask {
    /// Run all lints (fmt check + clippy)
    Lint,
    /// Run rustfmt
    Fmt {
        /// Check only, don't modify files
        #[arg(long)]
        check: bool,
    },
    /// Run clippy
    Clippy,
    /// Run tests
    Test,
    /// Run full CI suite (lint + test)
    Ci,
}

fn main() -> Result<()> {
    let cmd = Xtask::parse();
    match cmd {
        Xtask::Lint => {
            fmt(true)?;
            clippy()?;
        }
        Xtask::Fmt { check } => fmt(check)?,
        Xtask::Clippy => clippy()?,
        Xtask::Test => test()?,
        Xtask::Ci => {
            fmt(true)?;
            clippy()?;
            test()?;
        }
    }
    Ok(())
}

fn fmt(check: bool) -> Result<()> {
    let mut args = vec!["fmt", "--all"];
    if check {
        args.extend(["--", "--check"]);
    }
    run("cargo", &args).context("rustfmt failed")
}

fn clippy() -> Result<()> {
    run(
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    )
    .context("clippy failed")
}

fn test() -> Result<()> {
    run("cargo", &["test", "--workspace", "--all-targets"]).context("tests failed")
}

fn run(program: &str, args: &[&str]) -> Result<()> {
    eprintln!("==> {program} {}", args.join(" "));
    let status = Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("failed to run {program}"))?;
    if !status.success() {
        bail!("{program} exited with {status}");
    }
    Ok(())
}
