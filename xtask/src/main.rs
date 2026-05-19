use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser)]
#[command(name = "xtask", about = "rocket-surgeon build tasks")]
enum Xtask {
    /// Run all lints (Rust + Python)
    Lint,
    /// Run rustfmt
    Fmt {
        /// Check only, don't modify files
        #[arg(long)]
        check: bool,
    },
    /// Run clippy
    Clippy,
    /// Run Rust tests
    Test,
    /// Run ruff (Python linter + formatter check)
    Ruff {
        /// Fix issues automatically
        #[arg(long)]
        fix: bool,
    },
    /// Run mypy (Python type checker)
    Mypy,
    /// Run Python tests
    Pytest,
    /// Run TCK (Technology Compatibility Kit) tests
    Tck,
    /// Run full CI suite (all lints + all tests)
    Ci,
    /// Bootstrap the project (idempotent): venv, deps, maturin, cargo build
    Setup,
}

fn main() -> Result<()> {
    let cmd = Xtask::parse();
    match cmd {
        Xtask::Lint => {
            fmt(true)?;
            clippy()?;
            ruff(false)?;
            mypy()?;
        }
        Xtask::Fmt { check } => fmt(check)?,
        Xtask::Clippy => clippy()?,
        Xtask::Test => test()?,
        Xtask::Ruff { fix } => ruff(fix)?,
        Xtask::Mypy => mypy()?,
        Xtask::Pytest => pytest()?,
        Xtask::Tck => tck()?,
        Xtask::Ci => {
            fmt(true)?;
            clippy()?;
            ruff(false)?;
            mypy()?;
            test()?;
            pytest()?;
        }
        Xtask::Setup => setup()?,
    }
    Ok(())
}

fn setup() -> Result<()> {
    let repo_root = std::env::current_dir().context("cwd")?;
    let script = repo_root.join("scripts").join("bootstrap.sh");
    if !script.is_file() {
        bail!("bootstrap script not found at {}", script.display());
    }
    run("bash", &[script.to_str().context("non-utf8 script path")?]).context("bootstrap.sh failed")
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
    // PyO3 feature unification: rocket-surgeon-python uses `extension-module`
    // (suppresses libpython linking) while rocket-surgeon-worker uses
    // `auto-initialize` (requires libpython linking). Cargo unifies these
    // features when building the workspace, so the worker test binary fails
    // to link. Fix: test them separately.
    run(
        "cargo",
        &[
            "test",
            "--workspace",
            "--all-targets",
            "--exclude",
            "rocket-surgeon-worker",
        ],
    )
    .context("cargo test (workspace) failed")?;
    run_with_python_lib(
        "cargo",
        &["test", "-p", "rocket-surgeon-worker", "--all-targets"],
    )
    .context("cargo test (worker) failed")
}

fn ruff(fix: bool) -> Result<()> {
    if fix {
        run("ruff", &["check", "--fix", "python/", "tests/"])?;
        run("ruff", &["format", "python/", "tests/"]).context("ruff format failed")
    } else {
        run("ruff", &["check", "python/", "tests/"]).context("ruff check failed")?;
        run("ruff", &["format", "--check", "python/", "tests/"]).context("ruff format check failed")
    }
}

fn mypy() -> Result<()> {
    run("mypy", &["python/rocket_surgeon"]).context("mypy failed")
}

fn pytest() -> Result<()> {
    run("python3", &["-m", "pytest", "python/tests", "-v"]).context("pytest failed")
}

fn tck() -> Result<()> {
    run(
        "python3",
        &["-m", "pytest", "python/tests/tck", "-v", "--no-header"],
    )
    .context("tck tests failed")
}

fn python_libdir() -> Result<String> {
    let output = Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
        ])
        .output()
        .context("failed to query python3 LIBDIR")?;
    if !output.status.success() {
        bail!("python3 LIBDIR query failed");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Run a command with DYLD_LIBRARY_PATH / LD_LIBRARY_PATH set to the Python
/// shared library directory. Required for PyO3 `auto-initialize` binaries on
/// macOS where SIP strips DYLD vars from child processes.
fn run_with_python_lib(program: &str, args: &[&str]) -> Result<()> {
    let libdir = python_libdir()?;
    eprintln!(
        "==> DYLD_LIBRARY_PATH={libdir} {program} {}",
        args.join(" ")
    );
    let status = Command::new(program)
        .args(args)
        .env("DYLD_LIBRARY_PATH", &libdir)
        .env("LD_LIBRARY_PATH", &libdir)
        .status()
        .with_context(|| format!("failed to run {program}"))?;
    if !status.success() {
        bail!("{program} exited with {status}");
    }
    Ok(())
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
