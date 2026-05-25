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
    /// Run model conformance tests
    Conformance,
    /// Run full CI suite (all lints + all tests)
    Ci,
    /// Bootstrap the project (idempotent): venv, deps, maturin, cargo build
    Setup,
    /// Watch sources and rebuild the workspace on every change
    Watch,
    /// Watch sources and rerun tests on every change
    TestWatch {
        /// Optional substring; only e2e scripts whose filename contains it run.
        /// When omitted, the full Rust test suite plus every e2e script runs.
        pattern: Option<String>,
    },
    /// Internal: single rebuild iteration invoked by `watch`
    #[command(hide = true)]
    WatchOnce,
    /// Internal: single test iteration invoked by `test-watch`
    #[command(hide = true)]
    TestWatchOnce { pattern: Option<String> },
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
        Xtask::Conformance => conformance()?,
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
        Xtask::Watch => watch()?,
        Xtask::TestWatch { pattern } => test_watch(pattern)?,
        Xtask::WatchOnce => watch_once()?,
        Xtask::TestWatchOnce { pattern } => test_watch_once(pattern)?,
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

fn conformance() -> Result<()> {
    run(
        &venv_python()?,
        &[
            "-m",
            "pytest",
            "python/tests/conformance",
            "-v",
            "--no-header",
            "-m",
            "not nightly",
        ],
    )
    .context("conformance tests failed")
}

fn tck() -> Result<()> {
    run(
        &venv_python()?,
        &["-m", "pytest", "python/tests/tck", "-v", "--no-header"],
    )
    .context("tck tests failed")
}

fn ensure_cargo_watch() -> Result<()> {
    if Command::new("cargo-watch")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Ok(());
    }
    bail!("cargo-watch not found — run `cargo xtask setup` or `cargo install cargo-watch`");
}

fn watch() -> Result<()> {
    ensure_cargo_watch()?;
    run(
        "cargo",
        &[
            "watch",
            "--clear",
            "--watch",
            "crates",
            "--watch",
            "python",
            "--ext",
            "rs,py",
            "-x",
            "xtask watch-once",
        ],
    )
}

fn test_watch(pattern: Option<String>) -> Result<()> {
    ensure_cargo_watch()?;
    let inner = match pattern.as_deref() {
        Some(p) => format!("xtask test-watch-once {p}"),
        None => String::from("xtask test-watch-once"),
    };
    run(
        "cargo",
        &[
            "watch",
            "--clear",
            "--watch",
            "crates",
            "--watch",
            "python",
            "--watch",
            "tests",
            "--watch",
            "tck",
            "--ext",
            "rs,py,feature",
            "-x",
            &inner,
        ],
    )
}

/// Two-phase workspace build matching `tests/e2e_harness.py::build_binaries`.
/// PyO3 feature unification forces splitting `rocket-surgeon-worker` (which
/// uses `auto-initialize` and needs libpython) from the rest of the workspace
/// (where `extension-module` suppresses libpython linking).
fn build_all() -> Result<()> {
    run(
        "cargo",
        &[
            "build",
            "--workspace",
            "--exclude",
            "rocket-surgeon-python",
            "--exclude",
            "rocket-surgeon-worker",
        ],
    )
    .context("workspace build failed")?;

    let py = venv_python()?;
    let libdir = Command::new(&py)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
        ])
        .output()
        .context("query python LIBDIR")?;
    if !libdir.status.success() {
        bail!("failed to query venv LIBDIR");
    }
    let libdir = String::from_utf8(libdir.stdout)
        .context("non-utf8 LIBDIR")?
        .trim()
        .to_owned();

    eprintln!("==> cargo build -p rocket-surgeon-worker (DYLD/LD_LIBRARY_PATH={libdir})");
    let status = Command::new("cargo")
        .args(["build", "-p", "rocket-surgeon-worker"])
        .env("DYLD_LIBRARY_PATH", &libdir)
        .env("LD_LIBRARY_PATH", &libdir)
        .status()
        .context("failed to run cargo")?;
    if !status.success() {
        bail!("worker build exited with {status}");
    }
    Ok(())
}

fn now_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn watch_once() -> Result<()> {
    let stamp = now_stamp();
    eprintln!("[watch {stamp}] rebuild starting");
    match build_all() {
        Ok(()) => {
            let done = now_stamp();
            eprintln!("[watch {done}] rebuild OK");
            Ok(())
        }
        Err(e) => {
            let done = now_stamp();
            eprintln!("[watch {done}] rebuild FAILED: {e:#}");
            Err(e)
        }
    }
}

fn e2e_scripts_matching(pattern: Option<&str>) -> Result<Vec<std::path::PathBuf>> {
    let tests_dir = std::env::current_dir().context("cwd")?.join("tests");
    let mut scripts: Vec<std::path::PathBuf> = std::fs::read_dir(&tests_dir)
        .with_context(|| format!("read {}", tests_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.is_file()
                && path.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n.starts_with("test_e2e_")
                        && n.ends_with(".py")
                        && pattern.is_none_or(|p| n.contains(p))
                })
        })
        .collect();
    scripts.sort();
    Ok(scripts)
}

fn test_watch_once(pattern: Option<String>) -> Result<()> {
    let stamp = now_stamp();
    let label = pattern.as_deref().unwrap_or("<all>");
    eprintln!("[test-watch {stamp}] run starting (pattern={label})");

    let result = (|| -> Result<()> {
        if pattern.is_none() {
            test()?;
        }
        let scripts = e2e_scripts_matching(pattern.as_deref())?;
        if scripts.is_empty() {
            if let Some(p) = pattern.as_deref() {
                bail!("no e2e scripts matched pattern '{p}'");
            }
            bail!("no e2e scripts found");
        }
        let py = venv_python()?;
        let mut failures = Vec::new();
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
    })();

    let done = now_stamp();
    match &result {
        Ok(()) => eprintln!("[test-watch {done}] PASS (pattern={label})"),
        Err(e) => eprintln!("[test-watch {done}] FAIL (pattern={label}): {e:#}"),
    }
    result
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
