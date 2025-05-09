The final project code is in the folder "Final".

The necessary files to run the simulation are "ThreeBody.py" and "ThreeBody_lib.py"

All other files in the folder, aside from the pycache, are the example files spit out by the simulation process.

In the "ThreeBody.py" file, if the animate is set to True, an animated gif of the data will be returned, "ThreeBody.gif". Otherwise, the only result from the code will be the .dat files: "ThreeBody*n*.dat".

To run the simulation, you will need to have access to the imports:

 - "numpy"

 - "time"

 - "matplotlib", "pyplot" and "animation"

 - And run on python3.8

If all of these are true download "ThreeBody.py" and "ThreeBody_lib.py" into the same directory. In that directory, enter into Bash and execute the command "python ThreeBody.py". This will take a little while, but will result in the same files being created as are in the "Final" folder.

Those files are simply a test case. In order to run unique simulations, pay attention to the variables at the top of the script:

 - dt

 - T

 - safe_r

 - m1, m2, and m3

 - p_initial_1, p_initial_2, and p_initial_3

 - v_initial_1, v_initial_2, and v_initial_3

dt is the time step for the simulation, a smaller time step results in longer simulation time but smoother simulation. Units are seconds.

T is the total length of the simulation. Units are seconds

safe_r is the distance away from each body that the bodies will be forced away. If the bodies get too close, they can experience unnaturally high acceleration. This can be limited with small time steps, dt, or a protective "force field" around each body.

m1, m2, and m3 are the masses of the bodies. Units are kg. Values are recommended of 10^7 or higher magnitude.

p_initial_1, p_initial_2, and p_initial_3 are the initial positions of the bodies. Represented as matrices of the x, y, and z coordinates of the bodies. The coordinates are in m units. Issues can arise when the bodies are placed directly on top of each other, give them some space.

v_initial_1, v_initial_2, and v_initial_3 are the initial velocities of the bodies. Represented as matrices of the x, y, and z components of the velocities. The velocities are in m/s units. It is recommended to start with velocities an order of magnitude less than the magnitude of the initial positions.

Errors are sometimes expected. If an index error occurs around row 90 in the "ThreeBody.py" script, change the "safe_r" variable until you fin one that works for the situation.

