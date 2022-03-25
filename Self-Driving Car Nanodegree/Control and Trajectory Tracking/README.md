# Control and Trajectory Tracking
In this repository, you find the codes, written in C++,  for designng a PID controller to enable an autonomous vehicle to track reference trajectories. In this work, we control both throttle/brake pedal and steering at the same time. Given a trajectory as an array of locations, and a simulation environment (the vehicle with possible perturbations), you will design and code a PID controller and test its efficiency on the CARLA simulator used in the industry. 

## Installation
Run the following commands to install the starter code in the Udacity Workspace:

Clone the repository:

    git clone https://github.com/udacity/nd013-c6-control-starter.git

### Run Carla Simulator
Open new window

    su - student // Will say permission denied, ignore and continue
    cd /opt/carla-simulator/
    SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl

### Compile and Run the Controller
Open new window

    cd nd013-c6-control-starter/project
    ./install-ubuntu.sh
    cd pid_controller/
    rm -rf rpclib
    git clone https://github.com/rpclib/rpclib.git
    cmake .
    make (This last command compiles your C++ code, run it after every change in your code)

### Testing
To test your installation run the following commands.

    cd nd013-c6-control-starter/project
    ./run_main_pid.sh Go to desktop mode to see CARLA

If error bind is already in use, or address already being used
    ps -aux | grep carla
    kill id

### Tips
- When you will be testing your C++ code, restart the Carla simulator to remove the former car from the simulation.
- If the simulation freezes on the desktop mode but is still running on the terminal, close the desktop and restart it.

## Project Dependencies Overview
1- CARLA simulator in a workspace.
2- A C++ solver open sources and used in the industry
3- Code to interact with the CARLA simulator
