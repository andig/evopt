# evopt

[![Go Reference](https://pkg.go.dev/badge/github.com/andig/evopt.svg)](https://pkg.go.dev/github.com/andig/evopt)

Inspired by https://github.com/Akkudoktor-EOS/EOS/pull/462

### Example

Request:

```
┌──────┬──────────┬──────────────┬──────────────────┬──────────────────┬────────────┐
│ HOUR │ FORECAST │ TOTAL DEMAND │ GRID IMPORT COST │ GRID EXPORT COST │ BAT 0 GOAL │
├──────┼──────────┼──────────────┼──────────────────┼──────────────────┼────────────┤
│    1 │     2000 │         3000 │             0.30 │             0.15 │          - │
│    2 │     6000 │         4000 │             0.25 │             0.12 │          - │
│    3 │     8000 │         5000 │             0.20 │             0.10 │      40000 │
│    4 │     7000 │         4500 │             0.22 │             0.11 │          - │
│    5 │     4000 │         3500 │             0.28 │             0.14 │          - │
│    6 │     1000 │         3000 │             0.32 │             0.16 │          - │
└──────┴──────────┴──────────────┴──────────────────┴──────────────────┴────────────┘
```

Response:

```
┌──────┬─────────────┬─────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ HOUR │ GRID IMPORT │ GRID EXPORT │ BAT 0 CHA │ BAT 0 DIS │ BAT 0 SOC │ BAT 1 CHA │ BAT 1 DIS │ BAT 1 SOC │
├──────┼─────────────┼─────────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│    1 │        1516 │           - │      4316 │         - │     19100 │         - │      3800 │      1000 │
│    2 │        9000 │           - │     11000 │         - │     29550 │         - │         - │      1000 │
│    3 │       10216 │           - │     11000 │         - │     40000 │      2216 │        -0 │      3105 │
│    4 │        7526 │           - │     10026 │         - │     49525 │         - │         - │      3105 │
│    5 │           - │           - │       500 │         - │     50000 │         - │         - │      3105 │
│    6 │           - │           - │         - │         - │     50000 │         - │      2000 │      1000 │
└──────┴─────────────┴─────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

Visualization:

```
 100.0 ┤                                                   ╭───────────────────────────────────────────────
  91.2 ┤                                          ╭────────╯
  82.5 ┤                                  ╭───────╯
  73.8 ┤                         ╭────────╯
  65.0 ┤                 ╭───────╯
  56.2 ┤         ╭───────╯
  47.5 ┤ ╭───────╯
  38.8 ┼─╯                               ╭───────────────────────────────────────────────────╮
  30.0 ┤                          ╭──────╯                                                   ╰─────╮
  21.2 ┤                    ╭─────╯                                                                ╰──────╮
  12.5 ┼────────────────────╯                                                                             ╰
                                                 Optimization - SoC
                                             ■ Bat 1 SoC   ■ Bat 2 SoC
```
```
 11000 ┤                  ╭──────────────────────────╮
 10450 ┤                 ╭╯                   ╭╮     ╰──────────╮
  9900 ┤               ╭─╯           ╭────────╯╰───╮            ╰───╮
  9350 ┤              ╭╯     ╭───────╯             ╰───╮            ╰╮
  8800 ┤            ╭─╯   ╭──╯                         ╰────╮        ╰╮
  8250 ┤          ╭─╯    ╭╯                    ╭╮           ╰───╮     ╰╮
  7700 ┤         ╭╯    ╭─╯               ╭─────╯╰─────────╮     ╰──╮   ╰╮
  7150 ┤       ╭─╯    ╭╯            ╭────╯                ╰─────────╮╮  ╰╮
  6600 ┤     ╭─╯    ╭─╯        ╭────╯                               ╰──╮ ╰─╮
  6050 ┤    ╭╯     ╭╯     ╭────╯                                      ╰╰───╰╮
  5500 ┤  ╭─╯    ╭─╯   ╭──╯                                             ╰╮ ╰╰╮─╮
  4950 ┤ ╭╯     ╭╯   ╭─╯                                                 ╰╮  ╰╮╰──╮
  4400 ┼─╯    ╭─╯ ╭──╯                                                    ╰─╮ ╰╮  ╰───╮
  3850 ┼─╮   ╭╯╭──╯                                                         ╰╮ ╰╮     ╰───╮
  3300 ┤ ╰──╮╭─╯                                                             ╰─╮╰╮        ╰──╮
  2750 ┤  ╭─╰─╮                                                                ╰╮╰─╮         ╰───╮
  2200 ┼──╯   ╰──╮                          ╭─────╮                             ╰─╮╰╮            ╰──╮     ╭
  1650 ┼─╯       ╰──╮                   ╭───╯     ╰────╮                          ╰╮╰╮              ╭─────╯
  1100 ┤            ╰──╮           ╭────╯              ╰────╮                      ╰─╰╮        ╭────╯   ╰──
   550 ┤               ╰──╮   ╭────╯                        ╰───╮                    ╰╰──╭─────╯╮
     0 ┼──────────────────╰──────────────────────────────────────────────────────────────╯─────────────────
                                             Optimization - Power Flow

        ■ Grid Import   ■ Grid Export   ■ Forecast   ■ Bat 1 Charge Power   ■ Bat 1 Discharge Power
                                                     ■ Bat 2 Charge Power   ■ Bat 2 Discharge Power
```

### Development

To develop the evopt codebase, ensure `uv` and `make` are available on your system.
Installation instructions for `uv` [can be found here](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` and `make` are available on the PATH, you can run `make run` to set up the project environment and run the optimizer service.

Linting and formatting is run with `make lint`.
The test suite is run with `make test`.
To add a new dependency to the project, run `uv add <dependency>`.
To upgrade all depdendencies to their latest version, run `make upgrade`.

If you are using VSCode, we recommend the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [autopep8](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8), and [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extensions.
Set up `autopep8` as your formatter for Python files.
