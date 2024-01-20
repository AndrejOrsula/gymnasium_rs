# gymnasium_rs

<p align="left">
  <a href="https://crates.io/crates/gymnasium">                                        <img alt="crates.io"  src="https://img.shields.io/crates/v/gymnasium.svg"></a>
  <a href="https://docs.rs/gymnasium">                                                 <img alt="docs.rs"    src="https://docs.rs/gymnasium/badge.svg"></a>
  <!-- <a href="https://pypi.org/project/gymnasium_rs/">                                    <img alt="pypi.org"   src="https://img.shields.io/pypi/v/gymnasium_rs.svg"></a> -->
  <a href="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/rust.yml">   <img alt="Rust"       src="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/rust.yml/badge.svg"></a>
  <a href="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/python.yml"> <img alt="Python"     src="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/python.yml/badge.svg"></a>
  <a href="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/docker.yml"> <img alt="Docker"     src="https://github.com/AndrejOrsula/gymnasium_rs/actions/workflows/docker.yml/badge.svg"></a>
  <a href="https://deps.rs/repo/github/AndrejOrsula/gymnasium_rs">                     <img alt="deps.rs"    src="https://deps.rs/repo/github/AndrejOrsula/gymnasium_rs/status.svg"></a>
  <a href="https://codecov.io/gh/AndrejOrsula/gymnasium_rs">                           <img alt="codecov.io" src="https://codecov.io/gh/AndrejOrsula/gymnasium_rs/branch/main/graph/badge.svg"></a>
</p>

Rust implementation of [Gymnasium](https://gymnasium.farama.org) API for reinforcement learning. This implementation is compatible and interoperable with the [Python implementation](https://github.com/Farama-Foundation/Gymnasium).

## Overview

The workspace contains these packages:

- **[gymnasium](gymnasium):** Core library
- **[gymnasium_cli](gymnasium_cli):** CLI tool
- **[gymnasium_py](gymnasium_py):** Python module for interoperability with Rust environments

## Instructions

### <a href="#-rust"><img src="https://rustacean.net/assets/rustacean-flat-noshadow.svg" width="16" height="16"></a> Rust

Add `gymnasium` as a Rust dependency to your [`Cargo.toml`](https://doc.rust-lang.org/cargo/reference/manifest.html) manifest.

```toml
[dependencies]
gymnasium = "0.1"
```

<!-- **Examples:** [`gymnasium/examples/`](gymnasium/examples/) -->

<!-- ### <a href="#-python"><img src="https://www.svgrepo.com/show/354238/python.svg" width="16" height="16"></a> Python

> The Python module requires Rust and Cargo to compile extensions. Install them through your package manager or via <https://rustup.rs>.

Compile and install the `gymnasium_rs` Python module by installing this project with [`pip`](https://pypi.org/project/pip).

```bash
pip install git+https://github.com/AndrejOrsula/gymnasium_rs.git
```

**Examples:** [`gymnasium_py/examples/`](gymnasium_py/examples/) -->

### <a href="#-cli-tool"><img src="https://www.svgrepo.com/show/353478/bash-icon.svg" width="16" height="16"></a> CLI tool

Install the `gymnasium_rs` executable with `cargo`.

```bash
cargo install --locked gymnasium_cli
```

Afterwards, run the `gymnasium_rs` executable.

```bash
# Pass `--help` to show the usage and available options
gymnasium_rs
```

<details>
<summary><h3><a href="#-docker"><img src="https://www.svgrepo.com/show/448221/docker.svg" width="16" height="16"></a> Docker</h3></summary>

> To install [Docker](https://docs.docker.com/get-docker) on your system, you can run [`.docker/host/install_docker.bash`](.docker/host/install_docker.bash) to configure Docker with NVIDIA GPU support.
>
> ```bash
> .docker/host/install_docker.bash
> ```

#### Build Image

To build a new Docker image from [`Dockerfile`](Dockerfile), you can run [`.docker/build.bash`](.docker/build.bash) as shown below.

```bash
.docker/build.bash ${TAG:-latest} ${BUILD_ARGS}
```

#### Run Container

To run the Docker container, you can use [`.docker/run.bash`](.docker/run.bash) as shown below.

```bash
.docker/run.bash ${TAG:-latest} ${CMD}
```

#### Run Dev Container

To run the Docker container in a development mode (source code mounted as a volume), you can use [`.docker/dev.bash`](.docker/dev.bash) as shown below.

```bash
.docker/dev.bash ${TAG:-latest} ${CMD}
```

As an alternative, users familiar with [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) can modify the included [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) to their needs. For convenience, [`.devcontainer/open.bash`](.devcontainer/open.bash) script is available to open this repository as a Dev Container in VS Code.

```bash
.devcontainer/open.bash
```

#### Join Container

To join a running Docker container from another terminal, you can use [`.docker/join.bash`](.docker/join.bash) as shown below.

```bash
.docker/join.bash ${CMD:-bash}
```

</details>

## License

This project is dual-licensed to be compatible with the Rust project, under either the [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) licenses.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
