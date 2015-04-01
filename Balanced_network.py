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
J = 0.1*mV          # The EPSP amplitude
D = 1.5*ms          # The transmission delay

# Initialize neuron parameters
Tau_e = 20*ms       # Excitation period
Tau_rp = 2*ms       # Refractory period
Theta = 20*mV       # Threshold
Vr = 10*mV          # Reset value after threshold is reached

# TO BE DONE
eqs = """
dV/dt = (1-v)/tau : 1
"""

# Excitatory neuron group
group = NeuronGroup(Ne, eqs, threshold='V>Theta',
                    reset='V=Vr', refractory=Tau_rp)



print Ce, Ci

print 'Finished'