# Simulation of balanced network (Model A of Brunel (2000))
from brian2 import *

# Start simulation
print "start balanced network"

# Initialize network parameters
Ne = 10000          # Number of excitatory pyramidal cells
Ni = 2500           # Number of inhibitory cells
epsilon = 0.1       # Rate of sparseness
Ce = Ne*epsilon     # Number of connections receiving from excitatory neurons
Ci = Ni*epsilon     # Number of connections receiving from inhibitory neurons
Cext = Ce           # Number of connections receiving from external neurons
J = 0.1*mV          # The PSP amplitude
D = 1.5*ms          # The transmission delay

# # Initialize neuron parameters
Tau_e = 20*ms       # Excitation period
Tau_rp = 2*ms       # Refractory period
Theta = 20*mV       # Threshold
Vr = 10*mV          # Reset value after threshold is reached


muext = 2*Theta
g = 5
sigmaext = J*sqrt(2*Theta/J)

# # Leaky Integrate and Fire neuron equation. Not in use anymore
# eqs = """
# dv/dt = (El-v)/Tau_e : volt
# # El : volt
# Tau_e : second
# """

# Integrate and Fire neuron equation
eqs = """
dv/dt = (-v + muext + sigmaext * sqrt(Tau_e) * xi)/Tau_e : volt
Tau_e : second
"""

# Excitatory neuron group
group_e = NeuronGroup(2, eqs, threshold='v>Theta',
                    reset='v=Vr', refractory=Tau_rp)
# group.El = [30, 30]*mV
group_e.Tau_e = [20, 30]*ms
# group.v = Vr

group_i = NeuronGroup(2, eqs, threshold='v>Theta',
                    reset='v=Vr', refractory=Tau_rp)
# group.El = [30, 30]*mV
group_i.Tau_e = [20, 30]*ms
# group.v = Vr


S_e = Synapses(group_e, group_e, pre='v_post += J')
S_e.connect(0, 1)
S_e.delay = D

S_i = Synapses(group_i, group_i, pre='v_post += -g*J')
S_i.connect(0, 1)
S_i.delay = D

M_e = StateMonitor(group_e, 'v', record=True)
M_i= StateMonitor(group_i, 'v', record=True)
run(50*ms)


figure()



subplot(221)
title('Neuron 1 excitatory')
plot(M_e.t/ms, M_e.v[0])


subplot(223)
title('Neuron 2 excitatory')
plot(M_e.t/ms, M_e.v[1])


subplot(222)
title('Neuron 1 inhibitory')
plot(M_i.t/ms, M_i.v[0])


subplot(224)
title('Neuron 2 inhibitory')
plot(M_i.t/ms, M_i.v[1])

xlabel('Time (ms)')
ylabel('v')

show()

print 'Finished'