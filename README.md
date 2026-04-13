# bodyweight-exercise-pose-estimation

## Enviroment creation

conda env create -f environment.yml

## PoseVisualizer

For mediapipe:

$x_{px}, y_{px}$
$$x_{pixel} = \text{int}(x_{normalized} \cdot \text{image\_width})$$
$$y_{pixel} = \text{int}(y_{normalized} \cdot \text{image\_height})$$