# Simulation of balanced network with unsupervised learning.

from brian2 import *
import random

# Start simulation
print "start unsupervised learning"

# Initialize  parameters
dt = 0.1 * ms		# Timesteps of neurongroup
Tau_m = 20 * ms     # Membrane time constant
Vleak = -75 * mV    # Resting membrance potential
gleak = 10 * nS     # Leak conductance
gexc = 1 * nS       # Conductance of excitatory neurons
# ginh = 2 * nS       # Conductance of inhibitory neurons
# gext = 8 * nS
Eexc = 0 * mV       # Reversal potantial excitatory neurons
Einh = -80 * mV     # Reversal potantial inhbitory neurons
N_e = 800      # Number of excitatory input neurons (in paper 3600 used)
N_i = 200     # Number of inhibitory input neurons (in paper 900 used)
C_m = Tau_m * gleak #

Tau_rp = 5 * ms       # Refractory period
Theta = -50 * mV      # Threshold
Vr = -55 * mV         # Reset value after threshold is reached
tau_exc = 5 * ms
tau_inh = 10 * ms
epsilon = .1914893617 # in paper 0.05 used, but scaled for N = 1000 (Golomb 2000)
# epsilon = .05

Duration = 600 * ms


# Integrate and Fire neuron equation
eqs = """
dv/dt = ( gleak*(Vleak-v) + gexc*(Eexc-v) + ginh*(Einh-v)) / (C_m)  : volt
dgexc/dt = -gexc/tau_exc : siemens
dginh/dt = -ginh/tau_inh : siemens
"""

for a in arange(1.5,6.5,.5):
    gext = a * nS
    for b in arange(2,6.5,.5):
        ginh = b * nS
        
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
        
        
        
        # def visualise_total_connectivity(S_e, S_i, S_ei, S_ie):
        #     Ns = len(S_e.source) + len(S_i.source)
        #     Nt = len(S_e.target) + len(S_i.target)
        #     figure()
        #     newvar_i = numpy.append(numpy.append(numpy.append(S_e.i, S_i.i + len(S_e.source)), S_ei.i + len(S_e.source)), S_ie.i)
        #     newvar_j = numpy.append(numpy.append(numpy.append(S_e.j, S_i.j + len(S_e.source)), S_ei.j), S_ie.j + len(S_e.source))
        #     plot(newvar_i, newvar_j, 'ok')
        #     grid(which='both')
        #     xlim(-1, Ns)
        #     ylim(-1, Nt)
        #     xlabel('Source neuron index')
        #     ylabel('Target neuron index')
        
        # visualise_total_connectivity(S_e, S_i, S_ei, S_ie)
        
        
        
        
        # Function to determine rates for excitatory input neurons
        # def determineRates(N):
        #     Nu_i = []
        #     mu = random.randint(0,N-1)
        #     for i in range(N):
        #         
        #         # Equation of Yger 2013, materials & methods, Circular Gaussian simulations.
        #         # NB. Litteral implementation results in linear distributions. In order to
        #         # make them circular min(abs(i-mu), (N-abs(i-mu))) is used in stead of
        #         # abs(i-mu) which is stated in the paper. 
        #         nu = (50*(exp((float(min(abs(i-mu), (N-abs(i-mu))))**2)/20000)) + 5)
        #         Nu_i.append(nu)
        #     return Nu_i * Hz
        
        
        
        # # # # # Excitatory artificial input.
        # # # # input = PoissonGroup(1000, rates=determineRates(1000))
        # # # # 
        # # # # # Inhibitory artificial input
        # # # # input_inh = PoissonGroup(250, rates = F_i)
        # # # 
        # # # 
        # # # # # Excitatory input synapses. 
        # # # # S_e = Synapses(input, neurons,
        # # # #              '''w : siemens''',
        # # # #              pre='''gexc += w''',
        # # # #              connect=True
        # # # #              )
        # # # # for i in range(len(S_e.w)):
        # # # #     S_e.w[i] = random.gauss(gexc, gexc/3)
        # # # # 
        # # # # # Inhibitory input synapses. 
        # # # # S_i = Synapses(input_inh, neurons,
        # # # #              '''w : siemens''',
        # # # #              pre='''ginh += w''',
        # # # #              connect=True
        # # # #              )
        # # # # for i in range(len(S_i.w)):
        # # # #     S_i.w[i] = random.gauss(ginh, ginh/3)
        
        
        
        
        # Monitors. Only s_mon is used to visualize input neurons
        # and produces replica of Yger Fig 2A.
        # s_mon_e = SpikeMonitor(input)
        # s_mon_i = SpikeMonitor(input_inh)
        # # # # M = StateMonitor(neurons, 'v', record=[0, 10, 85, 90])
        # mon = StateMonitor(S_e, 'w', record=True)
        SM = SpikeMonitor(neurons)
        # PRM_e = PopulationRateMonitor(group_e)
        # PRM_i = PopulationRateMonitor(group_i)
        PRM = PopulationRateMonitor(neurons)
        
        
        # # Main loop to run simulation
        # for i in range(2):
            # # input.rates = determineRates(N_e)
            # # run(Duration/10, report='stdout')



        run(Duration, report='stdout')
        
        figname = 'fig' + str(int(10*a)) + str(int(10*b))
        fig = figure()
        # exec("%s = figure()"%(figname))
        print type(figname)
        print type(fig)
        randomsample = random.sample(xrange(N_e), 25)
        plotlist_t = []
        plotlist_i = []
        for n in range(len(SM.i)):
            if SM.i[n] in randomsample:
                plotlist_t.append(SM.t[n]/ms)
                plotlist_i.append(randomsample.index(SM.i[n]))
        
        subplot(311)
        title('gext = %s ginh = %s; 25 exc neurons; 25 inh neurons; global activity exc; inh'%(gext, ginh))
        ylabel('neuron index')
        plot(plotlist_t, plotlist_i, '.k')
        
        randomsample = random.sample(xrange(N_e, N_e + N_i), 25)
        plotlist_t = []
        plotlist_i = []
        for n in range(len(SM.i)):
            if SM.i[n] in randomsample:
                plotlist_t.append(SM.t[n]/ms)
                plotlist_i.append(randomsample.index(SM.i[n]))
        
        subplot(312)
        ylabel('neuron index')
        plot(plotlist_t, plotlist_i, '.k')
        
        subplot(313)
        ylabel('Frequency')
        xlabel('time (ms)')
        # plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
        plot(PRM.t/ms, PRM.rate)

        # 
        # subplot(414)
        # ylabel('Frequency')
        # xlabel('time (ms)')
        # # plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
        # plot(PRM_i.t/ms, PRM_i.rate)
        
        fig.savefig(figname + '.png', bbox_inches='tight')
        exec("%s = fig"%(figname))

        

# Plots of which most are not in use, but can be used for
# neuron and network visualization

# def visualise_connectivity(S):
	# Ns = len(S.source)
	# Nt = len(S.target)
	# figure(figsize=(10, 4))
	# subplot(121)
	# plot(zeros(Ns), arange(Ns), 'ok', ms=10)
	# plot(ones(Nt), arange(Nt), 'ok', ms=10)
	# for i, j in zip(S.i, S.j):
	# 	plot([0, 1], [i, j], '-k')
	# xticks([0, 1], ['Source', 'Target'])
	# ylabel('Neuron index')
	# xlim(-0.1, 1.1)
	# ylim(-1, max(Ns, Nt))
	# subplot(122)
	# plot(S.i, S.j, 'ok')
	# xlim(-1, Ns)
	# ylim(-1, Nt)
	# xlabel('Source neuron index')
	# ylabel('Target neuron index')


# visualise_connectivity(S_input)
# visualise_connectivity(S_i)


# # # V-t plots of 4 random excitatory and inhobitory single neurons
# figure()
# 
# subplot(221)
# title('Excitatory neuron')
# plot(M[0].t/ms, M[0].v)
# xlabel('Time (ms)')
# ylabel('v')
# 
# subplot(223)
# title('Excitatory neuron')
# plot(M[10].t/ms, M[10].v)
# xlabel('Time (ms)')
# ylabel('v')
# 
# subplot(222)
# title('Inhibitory neuron')
# plot(M[85].t/ms, M[85].v)
# xlabel('Time (ms)')
# ylabel('v')
# 
# subplot(224)
# title('Inhibitory neuron')
# plot(M[90].t/ms, M[90].v)
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

# # plot of excitatory input 
# figure()
# plot(s_mon_e.t/ms, s_mon_e.i, '.')
# 
# # plot of inhibitory input 
# figure()
# plot(s_mon_i.t/ms, s_mon_i.i, '.')

# 
# figure()
# plot(MS.t/ms, MS.i, '.')


# figure()
# hist(Nu_i)
#

# # plot for figure 2B
# figure()
# scatter(mon.t/ms, zeros(200))
# # , mon.w[0]


# show()


print 'Finished'