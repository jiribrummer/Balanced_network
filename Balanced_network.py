# Simulation of balanced network. Model of Brunel (2000).

from brian2 import *
import random

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
Ne = 1600         # Number of excitatory pyramidal cells
Ni = 400          # Number of inhibitory cells
epsilon = 0.4098    # Rate of sparseness. This value is calculated with N = 2000
# epsilon = 0.2577	# Rate of sparseness. This value is calculated with N = 4000	
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
dt = 0.1 * ms		# Timesteps of neurongroup

# # State A
# g = 3
# Vext = 2

# State B
g = 6
Vext = 4
# 
# # State C
# g = 5
# Vext = 2

# # State D
# g = 4.5
# Vext = 0.9

muext = Vext*Theta
sigmaext = J*sqrt(Vext*Theta/J)

Duration = 600*ms

# Integrate and Fire neuron equation
eqs = """
dv/dt = (-v + (muext + sigmaext * sqrt(Tau_e) * xi) )/Tau_e : volt 
"""

# Excitatory neuron group
neurons = NeuronGroup(Ne+Ni, eqs, dt=dt, threshold='v>Theta',
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
	
def visualise_total_connectivity(S_e, S_i, S_ei, S_ie):
	Ns = len(S_e.source) + len(S_i.source)
	Nt = len(S_e.target) + len(S_i.target)
	figure()
	newvar_i = numpy.append(numpy.append(numpy.append(S_e.i, S_i.i + len(S_e.source)), S_ei.i + len(S_e.source)), S_ie.i)
	newvar_j = numpy.append(numpy.append(numpy.append(S_e.j, S_i.j + len(S_e.source)), S_ei.j), S_ie.j + len(S_e.source))
	plot(newvar_i, newvar_j, 'ok')
	grid(which='both')
	xlim(-1, Ns)
	ylim(-1, Nt)
	xlabel('Source neuron index')
	ylabel('Target neuron index')

# # Uncomment for detaild view of connectivity
# visualise_connectivity(S_e)
# visualise_connectivity(S_i)
# visualise_connectivity(S_ei)
# visualise_connectivity(S_ie)

# # Total connectivity. Only useful when N is small
if Ne+Ni < 500:
	visualise_total_connectivity(S_e, S_i, S_ei, S_ie)

M_e = StateMonitor(group_e, 'v', record=True)
M_i= StateMonitor(group_i, 'v', record=True)

SM = SpikeMonitor(neurons)

PRM_e = PopulationRateMonitor(group_e)
PRM_i = PopulationRateMonitor(group_i)

run(Duration, report='stdout')


# # Figures

# # V-t plots of 4 random excitatory and inhobitory single neurons
figure(1)

excitatoryList = random.sample(xrange(0,Ne), 2)
inhibitoryList = random.sample(xrange(0,Ni), 2)

subplot(221)
title('Excitatory neuron: neuron %s'%(excitatoryList[0]))
plot(M_e.t/ms, M_e.v[excitatoryList[0]])

subplot(223)
title('Excitatory neuron: neuron %s'%(excitatoryList[1]))
plot(M_e.t/ms, M_e.v[excitatoryList[1]])

subplot(222)
title('Inhibitory neuron: neuron %s'%(inhibitoryList[0]))
plot(M_i.t/ms, M_i.v[inhibitoryList[0]])

subplot(224)
title('Inhibitory neuron: neuron %s'%(inhibitoryList[1]))
plot(M_i.t/ms, M_i.v[1])

xlabel('Time (ms)')
ylabel('v')


# # Plot of firing rate of all neurons, a random sample
# # of 50 neurons and a cumulative rate over time.
# figure(2)
# 
# subplot(311)
# title('Raster plot of the spikes of all neurons over time')
# 
# ylabel('neuron index')
# plot(SM.t/ms, SM.i, '.k')
# 
# 
# randomsample = random.sample(xrange(Ne+Ni), 50)
# plotlist_t = []
# plotlist_i = []
# for n in range(len(SM.i)):
# 	if SM.i[n] in randomsample:
# 		plotlist_t.append(SM.t[n]/ms)
# 		plotlist_i.append(randomsample.index(SM.i[n]))
# 
# subplot(312)
# title('Raster plot of the spikes of a random sample of 50 neurons over time')
# ylabel('neuron index')
# plot(plotlist_t, plotlist_i, '.k')
# 
# subplot(313)
# 
# title('Global activity of system over time')
# ylabel('Frequency')
# xlabel('time (ms)')
# # plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
# plot(PRM_e.t/ms, PRM_e.rate)


figure(2)



randomsample = random.sample(xrange(Ne+Ni), 50)
plotlist_t = []
plotlist_i = []
for n in range(len(SM.i)):
	if SM.i[n] in randomsample:
		plotlist_t.append(SM.t[n]/ms)
		plotlist_i.append(randomsample.index(SM.i[n]))

subplot(211)
title('Raster plot of the spikes of a random sample of 50 neurons over time')
ylabel('neuron index')
plot(plotlist_t, plotlist_i, '.k')

subplot(212)

title('Global activity of system over time')
ylabel('Frequency')
xlabel('time (ms)')
# plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
plot(PRM_e.t/ms, PRM_e.rate)


show()

print 'Finished'