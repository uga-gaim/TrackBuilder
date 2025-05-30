# Ship Track Reconstruction

## Overview

This project tackles the challenge of turning raw ship location data into full vessel trajectories, **especially when vital identification numbers like IMO or MMSI are missing**. Our `shiptrack_reconstructor` Python module infers these continuous ship tracks. It does this by analyzing how close individual data points are to each other in time and space, then uses other available details—like a ship's speed, course, or type—to make connections. In essence, this algorithm reconstructs ship trajectories from scattered, unidentified data points by linking them based on where and when they occurred, along with their associated attributes.

## Features

- Identifier-Agnostic Trajectory Reconstruction: Connects ship points into tracks even when unique vessel identifiers (like MMSI/IMO) aren't present.
- Spatiotemporal Linking: Uses geographic location and timestamps to figure out likely connections between data points.
- Attribute-Based Validation: Incorporates additional data attributes, such as changes in speed or heading, to confirm or refine point associations.
- Scalable Approach: Designed to efficiently process large volumes of Arctic shipping traffic data.

## Contributing

We welcome contributions! If you have suggestions for improvements, spot any bugs, or want to contribute code, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License—you can find more details in the [LICENSE](LICENSE) file.
