# Simulation of balanced network based on Yger

from brian2 import *
import random

# Start simulation
print "start Yger balance"

# Initialize  parameters
dt = 0.1 * ms		# Timesteps of neurongroup
Tau_m = 20 * ms     # Membrane time constant
Vleak = -75 * mV    # Resting membrance potential
gleak = 10 * nS     # Leak conductance
gexc = 1 * nS       # Conductance of excitatory neurons
ginh = 2 * nS       # Conductance of inhibitory neurons
gext = 5 * nS       # Conductance of external excitatory input
Eexc = 0 * mV       # Reversal potantial excitatory neurons
Einh = -80 * mV     # Reversal potantial inhbitory neurons
N_e = 8      # Number of excitatory input neurons (in paper 3600 used)
N_i = 2    # Number of inhibitory input neurons (in paper 900 used)
C_m = Tau_m * gleak #

Tau_rp = 5 * ms       # Refractory period
Theta = -50 * mV      # Threshold
Vr = -55 * mV         # Reset value after threshold is reached
tau_exc = 5 * ms
tau_inh = 10 * ms
epsilon = .1914893617 # in paper 0.05 used, but scaled for N = 1000 (Golomb 2000)
# epsilon = .05         # Sparseness for 4500 neurons

Duration = 40 * ms

# Integrate and Fire neuron equation
eqs = """
dv/dt = ( gleak*(Vleak-v) + gexc*(Eexc-v) + ginh*(Einh-v)) / (C_m)  : volt
dgexc/dt = -gexc/tau_exc : siemens
dginh/dt = -ginh/tau_inh : siemens
"""

# IF neurons. 
neurons = NeuronGroup(N_e+N_i, eqs, dt=dt, threshold='v>Theta',
                    reset='v=Vr', refractory=Tau_rp)
neurons.v = Vr

group_e = neurons[:N_e]
group_i = neurons[N_e:]


# # Input stimuli
P = PoissonGroup(N_e+N_i, 300 * Hz)

# # Connect input to neurons
S_input = Synapses(P, neurons,
             '''w : siemens''',
             pre='''gexc += w''')
for i in range(len(P)):
    S_input.connect(i,i)
for j in range(len(P)):
    S_input.w[j] = random.gauss(gext, gext/3)
    

# # Interconnect neurons
S_e = Synapses(group_e, group_e,
               '''w : siemens''',
               pre='gexc += w')
S_e.connect('i!=j', p=epsilon)
for i in range(len(S_e.delay)):
    S_e.delay[i] = (random.uniform(.1, 5.0) * ms)
for j in range(len(S_e.w)):
    S_e.w[j] = random.gauss(gexc, gexc/3)

S_ei = Synapses(group_i, group_e,
                '''w : siemens''',
                pre='ginh += w')
S_ei.connect('i!=j', p=epsilon)
for i in range(len(S_ei.delay)):
    S_ei.delay[i] = (random.uniform(.1, 5.0) * ms)
for j in range(len(S_ei.w)):
    S_ei.w[j] = random.gauss(ginh, ginh/3)

S_i = Synapses(group_i, group_i,
               '''w : siemens''',
               pre='ginh += w')
S_i.connect('i!=j', p=epsilon)
for i in range(len(S_i.delay)):
    S_i.delay[i] = (random.uniform(.1, 5.0) * ms)
for j in range(len(S_i.w)):
    S_i.w[j] = random.gauss(ginh, ginh/3)

S_ie = Synapses(group_e, group_i,
                '''w : siemens''',
                pre='gexc += w')
S_ie.connect('i!=j', p=epsilon)
for i in range(len(S_ie.delay)):
    S_ie.delay[i] = (random.uniform(.1, 5.0) * ms)
for j in range(len(S_ie.w)):
    S_ie.w[j] = random.gauss(gexc, gexc/3)



# # # # M = StateMonitor(neurons, 'v', record=[0, 10, 85, 90])
# mon = StateMonitor(S_e, 'w', record=True)
SM = SpikeMonitor(neurons)
# PRM_e = PopulationRateMonitor(group_e)
# PRM_i = PopulationRateMonitor(group_i)
PRM = PopulationRateMonitor(neurons)




run(Duration, report='stdout')



##############################################################################
## Adapted from https://groups.google.com/forum/#!searchin/briansupport/ISI/briansupport/xeF-w0aII8M/at47M2gkBe4J, 1-5-2015
##############################################################################


def get_trains_from_spikes(i, t, imax=None):
    if len(i):
        if imax is None:
            imax = amax(i)+1
        I = argsort(i)
        i = i[I]
        t = t[I]
        splitpoints, = hstack((diff(i),True)).nonzero()
        splitarray = split(t, splitpoints+1)
        trains = [array([]) for _ in xrange(imax)]
        for j, st in zip(i[splitpoints], splitarray):
            st.sort()
            trains[j] = st
        return trains
    else:
        if imax is None:
            imax = 1
        return [array([]) for _ in xrange(imax)]
    
    
def get_isi_from_trains(trains, flat=True):
    if flat:
        t = (2*len(trains)-1)*[nan]
        t[::2] = [sort(s) for s in trains]
        t = hstack(t)
        isi = diff(t)
        return isi[-isnan(isi)]
    else:
        return [diff(t) for t in trains]
    
def calculate_cv():
    # i = randint(10000, size=(1000000,))
    i = SM.i
    # print 'i', i
    # t = rand(len(i))
    t = SM.t/ms
    # print 't', t
    cvList = get_trains_from_spikes(i, t)
    isi_per_neuron = []
    # print isi_per_neuron
    cv_per_neuron = []
    emplist = [array([])]
    # print '1', cvList
    for n in range(len(cvList)):
        # print '2', n
        # print '2,1', mean(n)
        # print '2,2', type(n)
        # print 'emplist', emplist
        emplist[0] = cvList[n]
        # print emplist
        isi_of_neuron = get_isi_from_trains(emplist)
        # print '3', isi_of_neuron
        mean_isi = mean(isi_of_neuron)
        # print 'mean_isi: ', mean_isi
        # print type(mean_isi)
        if not math.isnan(mean_isi):
            std_isi = std(isi_of_neuron)
            # print 'std: ', std_isi
            CV_of_neuron = std_isi/mean_isi
            # print 'CV of neuron: ', CV_of_neuron
            isi_per_neuron.append(array(isi_of_neuron))
            cv_per_neuron.append(array(CV_of_neuron))
            # print '4', isi_per_neuron
            # print 'cv per neuron: ', cv_per_neuron
    
    # print 'cv per neuron: ', cv_per_neuron
    # print '5', isi_per_neuron
    # print 'cv per neuron: ', cv_per_neuron
    average_CV = mean(cv_per_neuron)
    print 'average_CV', average_CV
        

    # print 'isi', isi
    # print type(isi), type(cv_per_neuron)
    # hist(isi, 100)
    hist(cv_per_neuron, 20, normed = True)
    show()
   
calculate_cv()

##############################################################################

fig = figure()

subplot(311)
title('gext = %s ginh = %s; 25 exc neurons; 25 inh neurons; global activity exc; inh'%(gext, ginh))
ylabel('neuron index')
ylim((-.5,len(neurons)+.5))
plot(SM.t/ms, SM.i, '.k')

subplot(212)
ylabel('Frequency')
xlabel('time (ms)')
# plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
plot(PRM.t/ms, PRM.rate)

# subplot(414)
# ylabel('Frequency')
# xlabel('time (ms)')
# # plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
# plot(PRM_i.t/ms, PRM_i.rate)

# fig.savefig(figname + '.png', bbox_inches='tight')  # Save image in Jiri directory
exec("%s = fig"%('network'))

# figure()
# plot(MS.t/ms, MS.i, '.')

show()

print 'Finished'