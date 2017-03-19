import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

states_order = \
    ['theta',  #wheel rolling angle along floor
     'omega',  #Wheel rotational speed along floor
     'psi',    #Robot tilt
     'psidot', #Robot tilt rate (what the gyro measures)
    ]

def pack_states(state_dict):
    return [state_dict[key] for key in states_order]

def unpack_states(state_vector):
    return dict(zip(states_order, state_vector))


#derivation from http://publications.lib.chalmers.se/records/fulltext/163640.pdf
def state_derivatives(state_vector, t, params):
    state_dict = unpack_states(state_vector)
    theta = state_dict['theta']
    omega = state_dict['omega']
    psi = state_dict['psi']
    psidot = state_dict['psidot']

    m_base = params['m_base']
    m_top = params['m_top']
    L = params['L']
    r = params['r']


    M11 = (m_base + m_top) * r**2
    M12 = m_top * L * r * np.cos(psi)
    M22 = m_top * L**2
    M = np.array([[M11, M12], [M12, M22]])

    f_1 = m_top * L * r * psidot * np.sin(psi)
    f_2 = m_top * 9.81 * L * np.sin(psi)

    torque = 0 #TODO
    F = np.array([[f_1 + torque], [f_2 - torque]])

    ddtheta_ddpsi = np.linalg.solve(M, F)

    derivatives = {}
    derivatives['theta'] = omega
    derivatives['omega'] = ddtheta_ddpsi[0]
    derivatives['psi'] = psidot
    derivatives['psidot'] = ddtheta_ddpsi[1]

    return pack_states(derivatives)

init_states = \
    {'theta': 0.0,
     'omega': 0.0,
     'psi': 0.01,
     'psidot': 0.0,
    }
params = \
    {'m_base': 1.0,
     'm_top': 0.4,
     'L': 0.450,
     'r': 0.075/2,
    }

t = np.linspace(0, 10, 101)
sol = odeint(state_derivatives, pack_states(init_states), t, args=(params,))

plt.plot(t, sol[:, 0], label='theta(t)')
plt.plot(t, sol[:, 1], label='omega(t)')
plt.plot(t, sol[:, 2], label='psi(t)')
plt.plot(t, sol[:, 3], label='psidot(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show(block=False)