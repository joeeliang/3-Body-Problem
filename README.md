# Three-Body Simulation and Lyapunov Exponent Calculation

## Overview
This project simulates the motion of three bodies under gravitational attraction and calculates the proximity function to analyze the system's chaotic behavior. It combines the language of Python and C++ to create fast calculations and visualizations of the periodic trajectories that are possible.

## Visualization Controls
The proximity heatmap now includes an on-screen control panel. After clicking a point on the heatmap, use the buttons to manipulate the simulation:

- **Animate** – generate positions for the selected point and play a Pygame animation.
- **Loop** – run the looped animation from `data/cut_positions.csv`.
- **Enhance** – rerun the heatmap calculation for the current axis limits.
- **Zoom** – search the current view for the minimum proximity and zoom in.
- **Auto** – automatically perform repeated zoom and enhance steps.

Click anywhere on the heatmap to select the velocity pair before using the buttons.
