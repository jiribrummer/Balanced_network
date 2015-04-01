from brian2 import *

N = 5000
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = .1*second
C = 1000
sparseness = float(C)/N
J = .1*mV
muext = 25*mV
sigmaext = 1*mV

print tau

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

group = NeuronGroup(N, eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr
conn = Synapses(group, group, pre='V += -J',
                connect='rand()<sparseness', delay=delta)
M = SpikeMonitor(group)
LFP = PopulationRateMonitor(group)

run(duration)

figure()
subplot(211)
plot(M.t/ms, M.i, '.')
xlim(0, duration/ms)

subplot(212)
# Bin the rates (currently not implemented in PopulationRateMonitor directly)
window = 0.4*ms
window_length = int(window/defaultclock.dt)
cumsum = numpy.cumsum(numpy.insert(LFP.rate, 0, 0))
binned_rate = (cumsum[window_length:] - cumsum[:-window_length]) / window_length
plot(LFP.t[window_length-1:]/ms, binned_rate)
xlim(0, duration/ms)

show()