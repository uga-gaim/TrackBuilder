# Ship Track Reconstruction

## Overview

This project tackles the challenge of turning raw ship location data into full vessel trajectories, **especially when vital identification numbers like IMO or MMSI are missing**. Our `track_builder` Python module infers these continuous ship tracks. It does this by analyzing how close individual data points are to each other in time and space, then uses other available details—like a ship's speed, course, or type—to make connections. In essence, this algorithm reconstructs ship trajectories from scattered, unidentified data points by linking them based on where and when they occurred, along with their associated attributes.

## Features

- Identifier-Agnostic Trajectory Reconstruction: Connects ship points into tracks even when unique vessel identifiers (like MMSI/IMO) aren't present.
- Spatiotemporal Linking: Uses geographic location and timestamps to figure out likely connections between data points.
- Attribute-Based Validation: Incorporates additional data attributes, such as changes in speed or heading, to confirm or refine point associations.
- Scalable Approach: Designed to efficiently process large volumes of Arctic shipping traffic data.

## Installation

You can install the `track_builder` module directly from this repository.

#### First, clone the repository:

```bash
git clone git@github.com:uga-gaim/TrackBuilder.git
cd TrackBuilder
```

#### For Users

If you only want to use the module in your projects, install it using `pip`:

```bash
pip install .
```

#### For Developers

If you plan to contribute to the project, run tests, or use development tools like pytest, install the module along with its development dependencies:

```bash
# [Option] Manage a dedicated environment for dev
# conda create -n venv_ship python==3.12 -y
# conda activate venv_ship
# conda install -c conda-forge jupyterlab ipywidgets tqdm ipykernel -y

pip install -e ".[dev]"
```

## Contributing

We welcome contributions! If you have suggestions for improvements, spot any bugs, or want to contribute code, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License—you can find more details in the [LICENSE](LICENSE) file.
