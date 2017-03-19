import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

states_order = \
    [ 'theta'
    , 'omega'
    ]

def pack_states(state_dict):
    return [state_dict[key] for key in states_order]

def unpack_states(state_vector):
    return dict(zip(states_order, state_vector))

def state_derivatives(state_vector, t, params):
    state_dict = unpack_states(state_vector)
    theta = state_dict['theta']
    omega = state_dict['omega']

    b = params['b']
    c = params['c']

    derivatives = {}
    derivatives['theta'] = omega
    derivatives['omega'] = -b*omega - c*np.sin(theta)

    return pack_states(derivatives)

init_states = \
    { 'theta': np.pi - 0.1
    , 'omega': 0.0
    }
params = \
    { 'b':0.25
    , 'c':5.0
    }

t = np.linspace(0, 10, 101)
sol = odeint(state_derivatives, pack_states(init_states), t, args=(params,))

plt.plot(t, sol[:, 0], label='theta(t)')
plt.plot(t, sol[:, 1], label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show(block=False)