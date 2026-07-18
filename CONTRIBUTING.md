# Contribution guidelines

First off, thank you for considering contributing to corr-solver-rs.

If your contribution is not straightforward, please first discuss the change you
wish to make by creating a new issue before making the change.

## Reporting issues

Before reporting an issue on the
[issue tracker](https://github.com/luk036/corr-solver-rs/issues),
please check that it has not already been reported by searching for some related
keywords.

## Pull requests

Try to do one pull request per change.

### Updating the changelog

Update the changes you have made in
[CHANGELOG](https://github.com/luk036/corr-solver-rs/blob/main/CHANGELOG.md)
file under the **Unreleased** section.

Add the changes of your pull request to one of the following subsections,
depending on the types of changes defined by
[Keep a changelog](https://keepachangelog.com/en/1.0.0/):

- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

If the required subsection does not exist yet under **Unreleased**, create it!

## Developing

### Set up

This is no different than other Rust projects.

```shell
git clone https://github.com/luk036/corr-solver-rs
cd corr-solver-rs
cargo test
```

### Development Setup

#### Prerequisites

- Rust toolchain (stable or nightly)
- Git
- Editor with Rust support (VS Code, IntelliJ Rust plugin, etc.)

#### Recommended Tools

Install pre-commit hooks for automatic formatting and linting:
```bash
cargo install lefthook
lefthook install
```

Or manually run before committing:
```bash
cargo fmt --all
cargo clippy --all-targets --all-features --workspace
cargo test --all-features --workspace
```

#### Development Workflow

1. **Create a branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Follow code style
3. **Run tests locally**: `cargo test --all-features --workspace`
4. **Check formatting**: `cargo fmt --all -- --check`
5. **Run linter**: `cargo clippy --all-targets --all-features --workspace`
6. **Run benchmarks** (optional): `cargo bench`
7. **Commit with clear messages**: See [commit message guide](#commit-messages)
8. **Push and create PR**: Target `main` branch

### Useful Commands

- Build and run release version:

  ```shell
  cargo build --release && cargo run --release
  ```

- Run Clippy:

  ```shell
  cargo clippy --all-targets --all-features --workspace
  ```

- Run all tests:

  ```shell
  cargo test --all-features --workspace
  ```

- Check to see if there are code formatting issues

  ```shell
  cargo fmt --all -- --check
  ```

- Format the code in the project

  ```shell
  cargo fmt --all
  ```

- Run benchmarks:

  ```shell
  cargo bench
  ```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat: add LSQ correlation solver`
- `fix(qmi): correct gradient sign in QMI oracle`
- `docs: update README with quick start guide`
