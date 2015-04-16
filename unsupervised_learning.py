# Simulation of balanced network with unsupervised learning.

from brian2 import *
import random

# Start simulation
print "start unsupervised learning"

# Initialize  parameters
dt = 0.1 * ms		# Timesteps of neurongroup
Tau_m = 20 * ms     # Membrane time constant
Vleak = -75 * mV    # Resting membrance potential
Tdep = 40 * ms      # After depolarisation time constant
gleak = 10 * nS     # Leak conductance
gexc = 1 * nS       # Conductance of excitatory neurons
ginh = 10 * nS      # Conductance of inhibitory neurons
Idep = 50 * pA      # After depolarising current
Eexc = 0 * mV       # Reversal potantial excitatory neurons
Einh = -80 * mV     # Reversal potantial inhbitory neurons
N = 400             # Number of input neurons (in paper 1000 used)
F = 10 * Hz         # Frequency of inhibitory input neurons

Tau_rp = 5*ms       # Refractory period
Theta = -50*mV      # Threshold
Vr = -55*mV         # Reset value after threshold is reached

Duration = 100*ms

# Integrate and Fire neuron equation
# Still need to be fixed
eqs = """
dv/dt = ( (Vleak-v) ) / (Tau_m ): volt
# dIdep/dt = -Idep / Tdep : amp
"""

# Function to determine rates for excitatory input neurons
def determineRates(N):
    Nu_i = []
    mu = random.randint(0,N-1)
    for i in range(N):
        
        # Equation of Yger 2013, materials & methods, Circular Gaussian simulations.
        # NB. Litteral implementation results in linear distributions. In order to
        # make them circular min(abs(i-mu), (N-abs(i-mu))) is used in stead of
        # abs(i-mu) which is stated in the paper. 
        nu = (50*(exp((float(min(abs(i-mu), (N-abs(i-mu))))**2)/20000)) + 5)
        Nu_i.append(nu)
    return Nu_i * Hz
        
# IF neuron. Equation still to be fixed.
neuron = NeuronGroup(1, eqs, dt=dt, threshold='v>Theta',
					reset='v=Vr', refractory=Tau_rp)

# Excitatory artificial input.
input = PoissonGroup(N, rates=determineRates(N))

# Synapses. Not in use yet.
S = Synapses(input, neuron,
             '''w : 1''',
             connect=True,
             )
# S.w = 'rand() * gmax'
S.w = 1

# Monitors. Only s_mon is used to visualize input neurons
# and produces replica of Yger Fig 2A.
s_mon = SpikeMonitor(input)
# M= StateMonitor(neuron, 'v', record=True)
# mon = StateMonitor(S, 'w', record=[0, 1])
# MS = SpikeMonitor(neuron)


# Main loop to run simulation
for i in range(10):
    input.rates = determineRates(N)
    run(Duration, report='stdout')


# Plots of which most are not in use, but can be used for
# neuron and network visualization



# # V-t plots of 4 random excitatory and inhobitory single neurons
# figure()
# 
# title('Excitatory neuron')
# plot(M[0].t/ms, M[0].v)
# xlabel('Time (ms)')
# ylabel('v')

# figure()

# subplot(311)
# plot(S.w / gmax, '.k');
# ylabel('Weight / gmax')
# xlabel('Synapse index')
# subplot(312)
# hist(S.w / gmax, 20)
# xlabel('Weight / gmax')
# subplot(313)
# plot(mon.t/second, mon.w.T/gmax)
# xlabel('Time (s)')
# ylabel('Weight / gmax')
# tight_layout()

# plot of excitatory input 
figure()
plot(s_mon.t/ms, s_mon.i, '.')

# figure()
# subplot(211)
# plot(MS.t/ms, MS.i, '.')
#

# figure()
# hist(Nu_i)


show()


print 'Finished'