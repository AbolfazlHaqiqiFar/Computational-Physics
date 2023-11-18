import numpy as np
import matplotlib.pyplot as plt





def euler_method(h, m, k, x0, v0, num_steps):
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    times = np.zeros(num_steps)
    x = x0
    v = v0
    for i in range(num_steps):
        positions[i] = x
        velocities[i] = v
        times[i] = i * h

        a = -k / m * x
        v = v + a * h
        x = x + v * h

    return times, positions, velocities

def euler_method_plot(h, m, k, x0, v0, num_steps):
    times, positions, velocities = euler_method(h, m, k, x0, v0, num_steps)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, positions, label='Position')
    plt.title('Harmonic Oscillator: Position vs Time')
    plt.xlabel('Time')
    plt.ylabel('Position')

    plt.subplot(2, 1, 2)
    plt.plot(times, velocities, label='Velocity', color='orange')
    plt.title('Harmonic Oscillator: Velocity vs Time')
    plt.xlabel('Time')
    plt.ylabel('Velocity')

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------
def harmonic_oscillator(t, y, m, k):
    x, v = y
    dydt = [v, -k/m * x]
    return dydt

#-----------------------------------------------------------------------------------------------
def picard_method(h, N, m, k):
    t_values = np.zeros(N+1)
    x_values = np.zeros(N+1)
    v_values = np.zeros(N+1)

    # Initial conditions
    t_values[0] = 0
    x_values[0] = 1.0  # initial displacement
    v_values[0] = 0.0  # initial velocity

    for i in range(N):
        t = t_values[i]
        x = x_values[i]
        v = v_values[i]

        # Picard iteration
        x_next = x + h * v
        v_next = v - h * (k/m) * x

        # Update values
        t_values[i+1] = t + h
        x_values[i+1] = x_next
        v_values[i+1] = v_next

    return t_values, x_values, v_values

def picard_method_plot(h, N, m, k):
    t, x, v = picard_method(h, N, m, k)
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label='Displacement')
    plt.title('Harmonic Oscillator - Displacement vs Time')
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, v, label='Velocity', color='orange')
    plt.title('Harmonic Oscillator - Velocity vs Time')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()

    plt.tight_layout()
    return plt.show()

#-----------------------------------------------------------------------------------------------
def acceleration(x, v,b , k):
    return -b * v - k * x

def predictor_corrector(x, v, dt, b , k):
    # Predictor step
    x_pred = x + v * dt
    v_pred = v + acceleration(x, v, b , k) * dt

    # Corrector step
    x_corr = x + 0.5 * (v + v_pred) * dt
    v_corr = v + 0.5 * (acceleration(x, v, b, k) + acceleration(x_pred, v_pred, b , k)) * dt

    return x_corr, v_corr

def predictor_corrector_plot(x0 ,v0 , total_time, dt, b , k):
    
    # Initialize arrays to store results
    time_steps = np.arange(0.0, total_time, dt)
    positions = []
    velocities = []

    # Initial conditions
    x = x0
    v = v0

    # Time integration loop
    for t in time_steps:
        positions.append(x)
        velocities.append(v)

        # Use predictor-corrector method to update position and velocity
        x, v = predictor_corrector(x, v, dt, b , k)

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, positions, label='Position (x)')
    plt.title('Harmonic Oscillator: Position vs Time')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_steps, velocities, label='Velocity (v)', color='orange')
    plt.title('Harmonic Oscillator: Velocity vs Time')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()

    plt.tight_layout()
    return plt.show()