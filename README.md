# gymnasium_rs

<p align="left">
  <a href="https://crates.io/crates/gymnasium">                                        <img alt="crates.io" src="https://img.shields.io/crates/v/gymnasium.svg"></a>
  <!-- TODO[doc]: Enable shield once Python module is published on https://pypi.org -->
  <!-- <a href="https://pypi.org/project/gymnasium_rs/">                                    <img alt="PyPi"      src="https://img.shields.io/pypi/v/gymnasium_rs.svg"></a> -->
  <a href="https://docs.rs/gymnasium">                                                 <img alt="docs"      src="https://img.shields.io/docsrs/gymnasium.svg?label=docs.rs"></a>
  <a href="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/rust.yml">   <img alt="Rust"      src="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/rust.yml/badge.svg"></a>
  <a href="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/python.yml"> <img alt="Python"    src="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/python.yml/badge.svg"></a>
  <a href="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/docker.yml"> <img alt="Docker"    src="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/docker.yml/badge.svg"></a>
  <a href="https://deps.rs/repo/github/AndrejOrsula/gymnasium_rs">                     <img alt="deps"      src="https://deps.rs/repo/github/AndrejOrsula/gymnasium_rs/status.svg"></a>
  <a href="https://codecov.io/gh/AndrejOrsula/gymnasium_rs">                           <img alt="codecov"   src="https://codecov.io/gh/AndrejOrsula/gymnasium_rs/branch/main/graph/badge.svg"></a>
</p>

<!-- TODO[doc]: Update README.md -->

Gymnasium API for Reinforcement Learning.

- [Overview](#overview)
- [Instructions](#instructions)
  - [ Usage in Rust](#-usage-in-rust)
  - [ Usage in Python](#-usage-in-python)
  - [ CLI tool](#-cli-tool)
  - [ Examples, Tests, and Benchmarks](#-examples-tests-and-benchmarks)
  - [ Docker](#-docker)
    - [Build Image](#build-image)
    - [Run Container](#run-container)
    - [Run Dev Container](#run-dev-container)
    - [Join Container](#join-container)
- [License](#license)
- [Contributing](#contributing)

## Overview

The workspace contains these packages:

- **[gymnasium](gymnasium):** Core library
- **[gymnasium_cli](gymnasium_cli):** CLI tool
- **[gymnasium_py](gymnasium_py):** Python module

## Instructions

### <a href="#-usage-in-rust"><img src="https://rustacean.net/assets/rustacean-flat-noshadow.svg" width="16" height="16"></a> Usage in Rust

Add `gymnasium` as a Rust dependency to your [`Cargo.toml`](https://doc.rust-lang.org/cargo/reference/manifest.html) manifest.

<!-- TODO[doc]: Update Cargo.toml dependency once the package is published on https://crates.io -->

```toml
[dependencies]
gymnasium = { git = "https://github.com/AndrejOrsula/gymnasium_rs.git" }
```

**Examples:** [`gymnasium/examples/`](gymnasium/examples/)

### <a href="#-usage-in-python"><img src="https://www.svgrepo.com/show/354238/python.svg" width="16" height="16"></a> Usage in Python

> The Python module requires Rust and Cargo to compile extensions. Install them through your package manager or via <https://rustup.rs>.

Compile and install the `gymnasium` Python module by installing this project with [`pip`](https://pypi.org/project/pip).

```bash
pip3 install git+https://github.com/AndrejOrsula/gymnasium_rs.git
```

**Examples:** [`gymnasium_py/examples/`](gymnasium_py/examples/)

### <a href="#-cli-tool"><img src="https://www.svgrepo.com/show/353478/bash-icon.svg" width="16" height="16"></a> CLI tool

After cloning the repository, use `cargo` to run the `gymnasium_cli` binary.

```bash
# Pass `--help` to show the usage and available options
cargo run --release --bin gymnasium_cli -- --help
```

### <a href="#-examples-tests-and-benchmarks"><img src="https://www.svgrepo.com/show/269868/lab.svg" width="16" height="16"></a> Examples, Tests, and Benchmarks

[Rust examples](gymnasium/examples/) can be run with `cargo`. [Python examples](gymnasium_py/examples/) can be directly interpreted.

```bash
# Rust
cargo run --release --example ${EXAMPLE}
# Python
python3 gymnasium_py/examples/${EXAMPLE}.py
```

Tests and benchmarks can be run with `cargo`.

> Python tests require [`pytest`](https://docs.pytest.org/) (can be installed as an optional dependency with `pip3 install .[test]`).

```bash
# Tests (`--features python` enables Python tests with `pytest`)
cargo test --features python
# Benchmarks
cargo bench
```

### <a href="#-docker"><img src="https://www.svgrepo.com/show/448221/docker.svg" width="16" height="16"></a> Docker

> To install [Docker](https://docs.docker.com/get-docker) on your system, you can run [`install_docker.bash`](.docker/host/install_docker.bash) to configure Docker with NVIDIA GPU support (only for Debian-based distributions).
>
> ```bash
> .docker/host/install_docker.bash
> ```

#### Build Image

To build a new Docker image from [Dockerfile](Dockerfile), you can run [`build.bash`](.docker/build.bash) as shown below.

```bash
.docker/build.bash ${TAG:-latest} ${BUILD_ARGS}
```

#### Run Container

To run the Docker container, you can use [`run.bash`](.docker/run.bash) as shown below.

```bash
.docker/run.bash ${TAG:-latest} ${CMD}
```

#### Run Dev Container

To run the Docker container in a development mode (source code mounted as a volume), you can use [`dev.bash`](.docker/dev.bash) as shown below.

```bash
.docker/dev.bash ${TAG:-latest} ${CMD}
```

As an alternative for VS Code users familiar with [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers), you can modify the included [`devcontainer.json`](.devcontainer/devcontainer.json) to your needs.

#### Join Container

To join a running Docker container from another terminal, you can use [`join.bash`](.docker/join.bash) as shown below.

```bash
.docker/join.bash ${CMD:-bash}
```

## License

This project is dual-licensed to be compatible with the Rust project, under either the [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) licenses.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
