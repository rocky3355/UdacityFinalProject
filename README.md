# System Integration Project

The goal of this project was to incorporate several of the course's topics into one single project. A car will follow given line of points that are already available for the track. Gas, brake and steering have to be controlled accordingly in order to keep the car on the track. Camera images have to be used to detect traffic lights and their state. Depending on the state, the car has to come to a stop at the corresponding stop line (the positions of all stop lines on the track are known) and then start driving again, as soon as the traffic light switches to green.

## 1. Gas, brake and steering
For gas, a PI controller and for the brake, a simple P controller with following values are being used:
- Gas: P=0.1, I=0.02
- Brake: P=150

If the current velocity is below a certain threshold and the target velocity is 0, the maximum brake value of 1000 Nm is applied to ensure the stillstand of the car. For the steering, the provided yaw controller is being used. Multiplying the controller result by 2 is the only modification. Again, no steering will happen if the current velocity is below a certain threshold.

## 2. Traffic light detection
