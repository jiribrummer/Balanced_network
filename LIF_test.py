
from brian2 import *
import random

# # Initialize neuron parameters
Tau_e = 50*ms       # Excitation period
Tau_rp = 2*ms       # Refractory period
Theta = 20*mV       # Threshold
Vr = 10*mV          # Reset value after threshold is reached
dt = 0.1 * ms		# Timesteps of neurongroup
J = 0.1*mV          # The PSP amplitude


# State B
g = 6
Vext = 4

muext = Vext*Theta
sigmaext = J*sqrt(Vext*Theta/J)

# Integrate and Fire neuron equation
eqs = """
dv/dt = (-v + (muext + sigmaext * sqrt(Tau_e) * xi) )/Tau_e : volt 
"""

# Excitatory neuron group
neuron = NeuronGroup(1, eqs, dt=dt, threshold='v>Theta',
					reset='v=Vr', refractory=Tau_rp)

M_e = StateMonitor(neuron, 'v', record=True)
run(50 * ms, report='stdout')

figure(1)



title('tau = 50 ms')
plot(M_e.t/ms, M_e.v[0])
xlabel('time (ms)')

show()