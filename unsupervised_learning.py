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
gleak = 10 * nS
gexc = 1 * nS
ginh = 10 * nS
Idep = 50 * pA
Eexc = 0 * mV
Einh = -80 * mV
N = 200
F = 10 * Hz
Nu_i = []

Tau_rp = 5*ms       # Refractory period
Theta = -50*mV       # Threshold
Vr = -55*mV          # Reset value after threshold is reached

Duration = 100*ms

# Integrate and Fire neuron equation
eqs = """
dv/dt = ( (Vleak-v) ) / (Tau_m ): volt
# dIdep/dt = -Idep / Tdep : amp
"""

def determineRates(N, Nu_i):
    mu = random.randint(0,N-1)
    print 'mu', mu
    print 'range N', range(N)
    for i in range(N):
        # print 'in loop'
        # print 'i', i
        nu1 = ((float(i)-mu)**2)/20000
        nubijna2 = exp(nu1)
        Nu2 = (50*(nubijna2) + 5)
        # print 'nu1', nu1
        # print 'nubijna2', nubijna2
        # print 'nu2', Nu2
        # print 'nu i ', Nu_i
        Nu_i.append(Nu2)
        # print 'endloop'
    print 'nu i', Nu_i
    return Nu_i * Hz
        
    

# IF neuron
neuron = NeuronGroup(1, eqs, dt=dt, threshold='v>Theta',
					reset='v=Vr', refractory=Tau_rp)
input = PoissonGroup(N, rates=determineRates(N, Nu_i))
S = Synapses(input, neuron,
             '''w : 1''',
             connect=True,
             )
# S.w = 'rand() * gmax'
S.w = 1

M= StateMonitor(neuron, 'v', record=True)
mon = StateMonitor(S, 'w', record=[0, 1])
MS = SpikeMonitor(neuron)
s_mon = SpikeMonitor(input)

run(Duration, report='stdout')

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

figure()
plot(s_mon.t/ms, s_mon.i, '.')

figure()
subplot(211)
plot(MS.t/ms, MS.i, '.')

show()


print 'Finished'