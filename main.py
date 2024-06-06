import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, qr, solve_triangular, lu, ordqz

# System parameters
a1 = 1
b1 = 5

mass_cart = 0.5  + b1
mass_pole = 0.1  + 0.1 * a1
length_pole = 0.3
inertia_pole = 0.006
damping_coefficient = 0.1
gravity = 9.80665

# Simulation parameters
simulation_steps = 4000
time_step = 0.01


def calculate_system_matrices(m_cart, m_pole, l_pole, I_pole, b, g):
  """
  Calculates the state-space matrices (A, B) for the cart-pole system.
  """
  denominator = (m_cart + m_pole) * (I_pole + m_pole * l_pole**2) - (m_pole**2 * l_pole**2)
  a22 = -((I_pole + m_pole * l_pole**2) * b) / denominator
  a23 = -(m_pole**2 * g * l_pole**2) / denominator
  a42 = (m_pole * l_pole * b) / (denominator )
  a43 = ((m_pole * g * l_pole + m_pole**3 * g * l_pole**3) * denominator) / (denominator * (I_pole + m_pole * l_pole**2))
  b2 = (I_pole + m_pole * l_pole**2) / denominator
  b4 = -((m_pole * l_pole)) / (denominator)

  A = np.array([[0, 1, 0, 0],
                [0, a22, a23, 0],
                [0, 0, 0, 1],
                [0, a42, a43, 0]])
  B = np.array([[0], [b2], [0], [b4]])
  return A, B


def my_solve_continuous_are(A, B, Q, R):
    """
    Solves the continuous-time algebraic Riccati equation (CARE).
    """
    m, n = B.shape

    H = np.empty((2*m+n, 2*m+n))
    H[:m, :m] = A
    H[:m, m:2*m] = 0.
    H[:m, 2*m:] = B
    H[m:2*m, :m] = -Q
    H[m:2*m, m:2*m] = -A.conj().T
    H[m:2*m, 2*m:] = 0. 
    H[2*m:, :m] = 0.
    H[2*m:, m:2*m] = B.conj().T
    H[2*m:, 2*m:] = R


    J = np.block([np.eye((2 * m) + 1)])
    J[2*m][2*m] = R[0][0]

    q, r = qr(H[:, -n:])
    H = q[:, n:].conj().T.dot(H[:, :2*m])
    J = q[:2*m, n:].conj().T.dot(J[:2*m, :2*m])

    try:
      _, _, _, _, _, U = ordqz(H, J, sort='lhp', overwrite_a=True,
                          overwrite_b=True, check_finite=False,
                          output='real')
    except Exception as e:
       raise(e)


    U2 = U[:m, :m]
    U1 = U[m:, :m]
    up, ul, uu = lu(U2)
    X = solve_triangular(ul.conj().T,
                      solve_triangular(uu.conj().T,
                                      U1.conj().T,
                                      lower=True),
                      unit_diagonal=True,
                      ).conj().T.dot(up.conj().T)
    return X


def design_ackermann_controller(A, B):
  """
  Designs a controller using Ackermann's formula to place the closed-loop poles.
  """
  controllability_matrix = np.hstack((B, A @ B, A @ A @ B, A @ A @ A @ B))
  controllability_matrix_inv = inv(controllability_matrix)
  K_acker = np.array([0, 0, 0, 1]) @ controllability_matrix_inv @ np.linalg.matrix_power(A, 4) \
             + np.array([0, 0, 0, 1]) @ controllability_matrix_inv @ (5.5 * np.linalg.matrix_power(A, 3)) \
             + np.array([0, 0, 0, 1]) @ controllability_matrix_inv @ (11.1875 * np.linalg.matrix_power(A, 2)) \
             + np.array([0, 0, 0, 1]) @ controllability_matrix_inv @ (9.96875 * A + 3.28125)
  return K_acker


def design_lqr_controller(A, B, Q, R):
  """
  Designs a controller using Linear Quadratic Regulator (LQR) to minimize a cost function.
  """

  P = my_solve_continuous_are(A, B, Q, R)
  K_lqr = (inv(R) @ B.T @ P)[0]
  return K_lqr


def simulate_system(x_0, x_ref, A, B, C, K, num_steps, dt):
  """
  Simulates the cart-pole system with a given controller.
  """
  x = x_0
  y = [[] for _ in range(4)]
  for _ in range(num_steps):
    x_error = x_ref - x
    u = K @ x_error
    x = x + A @ x * dt + B * u * dt
    y[0].append(x[0, 0])
    y[1].append(x[1, 0])
    y[2].append(x[2, 0])
    y[3].append(x[3, 0])
  return y

def plot_system_output(y, title):
    """
    Plots the output of the cart-pole system.
    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)

    labels = [
        ('Przemieszczenie wózka', 'Droga [m]'),
        ('Prędkość wózka', 'Prędkość [m/s]'),
        ('Kąt wychylenia wahadła', 'Wychylenie wahadła [rad]'),
        ('Prędkość kątowa wahadła', 'Prędkość kątowa [rad/s]')
    ]

    for i, ax in enumerate(axs.flat):
        ax.set_title(labels[i][0])
        ax.plot(y[i])
        ax.grid()
        ax.set(xlabel='Czas [s]', ylabel=labels[i][1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Initialize system parameters
A, B = calculate_system_matrices(mass_cart, mass_pole, length_pole, inertia_pole, damping_coefficient, gravity)
C = np.eye(4)

# Initialize initial state and reference state
x_0 = np.array([[-10], [0], [0.05], [0]])
x_ref = np.array([[2], [0], [0], [0]])

# Design Ackermann controller
K_acker = design_ackermann_controller(A, B)

# Simulate system with Ackermann controller
y_acker = simulate_system(x_0, x_ref, A, B, C, K_acker, simulation_steps, time_step)
plot_system_output(y_acker, "Ackermann Controller")

# Design LQR controller
Q = np.array([[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]])
R = np.eye(1) * 1
K_lqr = design_lqr_controller(A, B, Q, R)


# Simulate system with LQR controller
y_lqr = simulate_system(x_0, x_ref, A, B, C, K_lqr, simulation_steps, time_step)
plot_system_output(y_lqr, "LQR Controller")