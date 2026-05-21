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
    /// Run end-to-end tests (spawn the daemon, drive the JSON-RPC protocol)
    E2e,
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
        Xtask::E2e => e2e()?,
        Xtask::Tck => tck()?,
        Xtask::Ci => {
            fmt(true)?;
            clippy()?;
            ruff(false)?;
            mypy()?;
            test()?;
            pytest()?;
            e2e()?;
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
    // Exclude PyO3 crates: rocket-surgeon-python is a cdylib whose test
    // binary needs libpython on LD_LIBRARY_PATH (unavailable in CI), and
    // rocket-surgeon-worker uses `auto-initialize` which conflicts with
    // extension-module under feature unification. Both are exercised by
    // pytest and e2e instead.
    run(
        "cargo",
        &[
            "test",
            "--workspace",
            "--all-targets",
            "--exclude",
            "rocket-surgeon-python",
            "--exclude",
            "rocket-surgeon-worker",
        ],
    )
    .context("cargo test failed")
}

fn ruff(fix: bool) -> Result<()> {
    let ruff = venv_bin("ruff")?;
    if fix {
        run(&ruff, &["check", "--fix", "python/", "tests/"])?;
        run(&ruff, &["format", "python/", "tests/"]).context("ruff format failed")
    } else {
        run(&ruff, &["check", "python/", "tests/"]).context("ruff check failed")?;
        run(&ruff, &["format", "--check", "python/", "tests/"]).context("ruff format check failed")
    }
}

fn mypy() -> Result<()> {
    run(&venv_bin("mypy")?, &["python/rocket_surgeon"]).context("mypy failed")
}

fn pytest() -> Result<()> {
    run(&venv_python()?, &["-m", "pytest", "python/tests", "-v"]).context("pytest failed")
}

/// Run every `tests/test_e2e_*.py` script. Each script builds the workspace
/// binaries and sets its own child-process environment, so the recipe just
/// invokes them. All scripts run even if one fails, so a single push surfaces
/// every regression at once.
fn e2e() -> Result<()> {
    let tests_dir = std::env::current_dir().context("cwd")?.join("tests");
    let mut scripts: Vec<std::path::PathBuf> = std::fs::read_dir(&tests_dir)
        .with_context(|| format!("read {}", tests_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("test_e2e_") && n.ends_with(".py"))
        })
        .collect();
    if scripts.is_empty() {
        bail!("no e2e test scripts found in {}", tests_dir.display());
    }
    scripts.sort();

    let mut failures = Vec::new();
    let py = venv_python()?;
    for script in &scripts {
        let path = script.to_str().context("non-utf8 script path")?;
        if run(&py, &["-u", path]).is_err() {
            failures.push(path.to_owned());
        }
    }
    if !failures.is_empty() {
        bail!(
            "{} e2e test(s) failed: {}",
            failures.len(),
            failures.join(", ")
        );
    }
    Ok(())
}

fn tck() -> Result<()> {
    run(
        &venv_python()?,
        &["-m", "pytest", "python/tests/tck", "-v", "--no-header"],
    )
    .context("tck tests failed")
}

/// Absolute path to an executable in the project virtualenv's `bin/`.
///
/// xtask is invoked from the repo root (guaranteed by the `cargo xtask`
/// alias), so the venv is always at `./.venv`. Calling venv executables
/// directly means tasks behave identically whether or not the venv is
/// activated in the calling shell — and always match `.python-version`.
fn venv_bin(name: &str) -> Result<String> {
    let exe = std::env::current_dir()
        .context("cwd")?
        .join(".venv/bin")
        .join(name);
    if !exe.is_file() {
        bail!(
            "{name} not found at {} — run `cargo xtask setup`",
            exe.display()
        );
    }
    exe.into_os_string()
        .into_string()
        .map_err(|_| anyhow::anyhow!("non-utf8 venv executable path"))
}

/// Absolute path to the project virtualenv's Python interpreter.
fn venv_python() -> Result<String> {
    venv_bin("python")
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
