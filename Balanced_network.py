# Simulation of balanced network (Model A of Brunel (2000))
from brian2 import *

# Start simulation
print "start balanced network"

# # # # # # Initialize network parameters
# # # # # Ne = 10000          # Number of excitatory pyramidal cells
# # # # # Ni = 2500           # Number of inhibitory cells
# # # # # epsilon = 0.1       # Rate of sparseness
# # # # # Ce = Ne*epsilon     # Number of connections receiving from excitatory neurons
# # # # # Ci = Ni*epsilon     # Number of connections receiving from inhibitory neurons
# # # # # Cext = Ce           # Number of connections receiving from external neurons
# # # # # J = 0.1*mV          # The PSP amplitude
# # # # # D = 1.5*ms          # The transmission delay
# # # # # 
# # # # # # # Initialize neuron parameters
# # # # # Tau_e = 20*ms       # Excitation period
# # # # # Tau_rp = 2*ms       # Refractory period
# # # # # Theta = 20*mV       # Threshold
# # # # # Vr = 10*mV          # Reset value after threshold is reached
# # # # # 
# # # # # 
# # # # # muext = 2*Theta
# # # # # g = 5
# # # # # sigmaext = J*sqrt(2*Theta/J)
# # # # # 
# # # # # Duration = 100*ms

# Initialize dummy network parameters
Ne = 1000          # Number of excitatory pyramidal cells
Ni = 250          # Number of inhibitory cells
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

g = 4.5
Vext = 0.9
muext = Vext*Theta
sigmaext = J*sqrt(Vext*Theta/J)

print muext
print sigmaext

Duration = 100*ms

# Integrate and Fire neuron equation
eqs = """
dv/dt = (-v + Ce*(muext + sigmaext * sqrt(Tau_e) * xi) )/Tau_e : volt 
"""

# Excitatory neuron group
neurons = NeuronGroup(Ne+Ni, eqs, threshold='v>Theta',
					reset='v=Vr', refractory=Tau_rp)

group_e = neurons[:Ne]
group_i = neurons[Ne:]


S_e = Synapses(group_e, group_e, pre='v_post += J')
S_e.connect('i!=j', p=epsilon)
S_e.delay = D

S_ei = Synapses(group_i, group_e, pre='v_post += -g*J')
S_ei.connect('i!=j', p=epsilon)
S_ei.delay = D

S_i = Synapses(group_i, group_i, pre='v_post += -g*J')
S_i.connect('i!=j', p=epsilon)
S_i.delay = D

S_ie = Synapses(group_e, group_i, pre='v_post += J')
S_ie.connect('i!=j', p=epsilon)
S_ie.delay = D


def visualise_connectivity(S_e):
	Ns = len(S_e.source)
	Nt = len(S_e.target)
	figure(figsize=(10, 4))
	subplot(121)
	plot(zeros(Ns), arange(Ns), 'ok', ms=10)
	plot(ones(Nt), arange(Nt), 'ok', ms=10)
	for i, j in zip(S_e.i, S_e.j):
		plot([0, 1], [i, j], '-k')
	xticks([0, 1], ['Source', 'Target'])
	ylabel('Neuron index')
	xlim(-0.1, 1.1)
	ylim(-1, max(Ns, Nt))
	subplot(122)
	plot(S_e.i, S_e.j, 'ok')
	xlim(-1, Ns)
	ylim(-1, Nt)
	xlabel('Source neuron index')
	ylabel('Target neuron index')

visualise_connectivity(S_e)
visualise_connectivity(S_i)
visualise_connectivity(S_ei)
visualise_connectivity(S_ie)

M_e = StateMonitor(group_e, 'v', record=True)
M_i= StateMonitor(group_i, 'v', record=True)

SM_e = SpikeMonitor(group_e[:50])
SM_i = SpikeMonitor(group_i[:50])

PRM_e = PopulationRateMonitor(group_e)
PRM_i = PopulationRateMonitor(group_i)

run(Duration, report='stdout')

print SM_e.num_spikes
print SM_i.num_spikes


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

figure()

subplot(211)
title('Excitatory')
plot(SM_e.t/ms, SM_e.i, '.k')

subplot(212)
# plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
plot(PRM_e.t/ms, PRM_e.rate)

figure()

subplot(211)
title('Inhibitory')
plot(SM_i.t/ms, SM_i.i, '.k')

subplot(212)
# plot(PRM_i.t/ms, PRM_i.rate*Ni/10000)
plot(PRM_i.t/ms, PRM_i.rate)

# subplot(212)
# title('Inhibitory')
# plot(SM_i.t/ms, SM_i.i, '.k')

xlabel('Time (ms)')
ylabel('Neuron index')


show()

print 'Finished'