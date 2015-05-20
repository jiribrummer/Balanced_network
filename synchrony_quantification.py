# Simulation of balanced network with unsupervised learning.

from brian2 import *
import random
import numpy.ma as ma
import matplotlib as mpl
import scipy.io as sio

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
N_e = 100     # Number of excitatory input neurons (in paper 3600 used)
N_i = 25     # Number of inhibitory input neurons (in paper 900 used)
C_m = Tau_m * gleak #

Tau_rp = 5 * ms       # Refractory period
Theta = -50 * mV      # Threshold
Vr = -55 * mV         # Reset value after threshold is reached
tau_exc = 5 * ms
tau_inh = 10 * ms
# epsilon = .1914893617 # in paper 0.05 used, but scaled for N = 1000 (Golomb 2000)
epsilon = .05

Duration = 600 * ms


# Integrate and Fire neuron equation
eqs = """
dv/dt = ( gleak*(Vleak-v) + gexc*(Eexc-v) + ginh*(Einh-v)) / (C_m)  : volt
dgexc/dt = -gexc/tau_exc : siemens
dginh/dt = -ginh/tau_inh : siemens
"""

ztemp = []              # Matrix where CV values will be stored in
yvalues = []            # List where gext values will be stored in
xvalues = []            # List where ginh values will be stored in
kmeans_input = []

gext_lower = 2         # Lower bound of gext for loop
gext_upper = 4        # Upper bound of gext for loop

ginh_lower = 2             # !!!!!!!!!!!!!!TO CHANGE: number of neurons and epsilon for large simulation !!!!!!!!!!!!!!!!!!!
ginh_upper = 4

stepsize = 1

for i in arange(gext_lower, gext_upper+stepsize, stepsize):
    yvalues.append(i)   # Add gext value to y-axis list

for j in arange(ginh_lower, ginh_upper+stepsize, stepsize):
    xvalues.append(j)   # Add ginh value to x-axis list


for a in arange(gext_lower,gext_upper,stepsize):
    ztemp.append([])    # Add row to matrix
    
    gext = a * nS
    
    for b in arange(ginh_lower,ginh_upper,stepsize):
        
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
            
            
        SM = SpikeMonitor(neurons)
        PRM = PopulationRateMonitor(neurons)

        run(Duration, report='stdout')
        
        figname1 = 'fig' + str(int(10*a)) + str(int(10*b))
        fig1 = figure()
        
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
        
        fig1.savefig(figname1 + '.png', bbox_inches='tight')
        
        scaled_PRMrate = PRM.rate/sum(PRM.rate)
        
        figname2 = 'ScaledHist' + str(int(10*a)) + str(int(10*b))
        fig2 = figure()
        hist(scaled_PRMrate, 50)
        fig2.savefig(figname2 + '.png', bbox_inches='tight')
        
        # figure()
        # hist(PRM.rate, 50)
        
        stripped_PRMrate = PRM.rate[PRM.rate != 0]
        stripped_scaled_PRMrate = stripped_PRMrate/sum(stripped_PRMrate)

        figname4 = 'StrippedScaledHist' + str(int(10*a)) + str(int(10*b))
        fig4 = figure()
        hist(stripped_scaled_PRMrate, 50)
        fig4.savefig(figname4 + '.png', bbox_inches='tight')
        
        frequency_measure2 = mean(stripped_scaled_PRMrate)
        print frequency_measure2
        
        ztemp[int((a-float(gext_lower))/stepsize)].append(frequency_measure2)
        kmeans_input.append(frequency_measure2)
        
zvalues = ma.array(ztemp, mask=np.isnan(ztemp))
# print zvalues

colorplot = figure()
title('Measure of synchrony of different gext and ginh values')
xlabel('ginh (nS)')
ylabel('gext (nS)')
colormap = mpl.cm.RdBu
colormap.set_bad('k', 1.)
pcolormesh(array(xvalues), array(yvalues), zvalues, cmap = colormap)
colorbar()

colorplot.savefig('colorplot.png', bbox_inches='tight')

sio.savemat('Matlab_colorplotData.mat', {'zvalues':ztemp, 'xvalues':xvalues, 'yvalues':yvalues})
sio.savemat('Matlab_kmeans.mat', {'kmeansvalues':kmeans_input})

        
print 'Finished'