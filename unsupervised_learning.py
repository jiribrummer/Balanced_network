# Simulation of balanced network with unsupervised learning.

from brian2 import *
import random

# Start simulation
print "start unsupervised learning"

# Initialize  parameters
dt = 0.1 * ms		# Timesteps of neurongroup
Tau_m = 20 * ms     # Membrane time constant
Vleak = -75 * mV    # Resting membrance potential
tau_dep = 40 * ms      # After depolarisation time constant
gleak = 10 * nS     # Leak conductance
gexc = 1 * nS       # Conductance of excitatory neurons
ginh = 10 * nS      # Conductance of inhibitory neurons
Idep = 50 * nS      # After depolarising current
Eexc = 0 * mV       # Reversal potantial excitatory neurons
Einh = -80 * mV     # Reversal potantial inhbitory neurons
N_e = 8            # Number of excitatory input neurons (in paper 1000 used)
N_i = 2           # Number of inhibitory input neurons (in paper 250 used)
F_i = 10 * Hz       # Frequency of inhibitory input neurons
C_m = Tau_m * gleak #

Tau_rp = 5*ms       # Refractory period
Theta = -50*mV      # Threshold
Vr = -55*mV         # Reset value after threshold is reached
tau_exc = 5 * ms
tau_inh = 10 * ms

Duration = 100*ms

# Integrate and Fire neuron equation
# Still need to be fixed
eqs = """
dv/dt = ( gleak*(Vleak-v) + gexc*(Eexc-v) + ginh*(Einh-v)) / (C_m): volt
dgexc/dt = -gexc/tau_exc : siemens
dginh/dt = -ginh/tau_inh : siemens
# dIdep/dt = -Idep/tau_dep : amp
"""

# # Learning rule
v = -69 
# F = -1 / (1 + exp(-(v+55 * mV)/(4 * mV))) + .5 * (2 * mV) * log(1 + exp((v+(52 * mV )/(2 * mV))))
F = (-1 / (1 + exp(-((v+55)/(4))))) + .5 * 2 * log10(1 + exp((v+52)/(2)))



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
neuron.v = Vr

# Excitatory artificial input.
input = PoissonGroup(N_e, rates=determineRates(N_e))

# Inhibitory artificial input
input_inh = PoissonGroup(N_i, rates = F_i)


# Excitatory synapses. 
S_e = Synapses(input, neuron,
             '''w : siemens''',
             pre='''gexc += w''',
             connect=True
             )
for i in range(len(S_e.w)):
    S_e.w[i] = random.gauss(gexc, gexc/3)

# Inhibitory synapses. 
S_i = Synapses(input_inh, neuron,
             '''w : siemens''',
             pre='''ginh += w''',
             connect=True
             )
for i in range(len(S_i.w)):
    S_i.w[i] = random.gauss(ginh, ginh/3)

# Monitors. Only s_mon is used to visualize input neurons
# and produces replica of Yger Fig 2A.
s_mon_e = SpikeMonitor(input)
s_mon_i = SpikeMonitor(input_inh)
M= StateMonitor(neuron, 'v', record=True)
mon = StateMonitor(S_e, 'w', record=True)
MS = SpikeMonitor(neuron)


# Main loop to run simulation
for i in range(2):
    input.rates = determineRates(N_e)
    run(Duration/10, report='stdout')




# Plots of which most are not in use, but can be used for
# neuron and network visualization

def visualise_connectivity(S):
	Ns = len(S.source)
	Nt = len(S.target)
	figure(figsize=(10, 4))
	subplot(121)
	plot(zeros(Ns), arange(Ns), 'ok', ms=10)
	plot(ones(Nt), arange(Nt), 'ok', ms=10)
	for i, j in zip(S.i, S.j):
		plot([0, 1], [i, j], '-k')
	xticks([0, 1], ['Source', 'Target'])
	ylabel('Neuron index')
	xlim(-0.1, 1.1)
	ylim(-1, max(Ns, Nt))
	subplot(122)
	plot(S.i, S.j, 'ok')
	xlim(-1, Ns)
	ylim(-1, Nt)
	xlabel('Source neuron index')
	ylabel('Target neuron index')


visualise_connectivity(S_e)
visualise_connectivity(S_i)


# # V-t plots of 4 random excitatory and inhobitory single neurons
figure()

title('Excitatory neuron')
plot(M[0].t/ms, M[0].v)
xlabel('Time (ms)')
ylabel('v')

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
plot(s_mon_e.t/ms, s_mon_e.i, '.')

# plot of inhibitory input 
figure()
plot(s_mon_i.t/ms, s_mon_i.i, '.')

# 
# figure()
# plot(MS.t/ms, MS.i, '.')


# figure()
# hist(Nu_i)
#

# plot for figure 2B
figure()
scatter(mon.t/ms, zeros(200))
# , mon.w[0]


show()


print 'Finished'