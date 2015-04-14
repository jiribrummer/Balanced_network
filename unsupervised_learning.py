# Simulation of balanced network with unsupervised learning.

from brian2 import *
import random

# Start simulation
print "start unsupervised learning"

# Initialize  parameters
dt = 0.1 * ms		# Timesteps of neurongroup
Tau_m = 20 * ms       # Excitation period
Vleak = -75 * mV
Tdep = 40 * ms
gleak = 10 
gexc = 1 * nS
ginh = 10 * nS
Idep = 50 * pA
Eexc = 0 * mV
Einh = -80 * mV

Tau_rp = 5*ms       # Refractory period
Theta = -50*mV       # Threshold
Vr = -55*mV          # Reset value after threshold is reached

Duration = 200*ms

# Integrate and Fire neuron equation
eqs = """
dv/dt = ( gleak*(Vleak-v) ) / Tau_m : volt
# dIdep/dt = -Idep / Tdep : amp
"""

# IF neuron
neuron = NeuronGroup(1, eqs, dt=dt, threshold='v>Theta',
					reset='v=Vr', refractory=Tau_rp)


M= StateMonitor(neuron, 'v', record=True)

run(Duration, report='stdout')

# # V-t plots of 4 random excitatory and inhobitory single neurons
figure(1)

title('Excitatory neuron')
plot(M[0].t/ms, M[0].v)
xlabel('Time (ms)')
ylabel('v')
show()

print 'Finished'