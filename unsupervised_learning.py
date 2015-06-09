# Simulation of balanced network with unsupervised learning.

from brian2 import *
import random
import numpy.ma as ma
import matplotlib as mpl
import scipy.io as sio
import time

# Start simulation
print "start unsupervised learning"
start = time.time()
print start


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

record_duration = 700*ms

# Integrate and Fire neuron equation
eqs = """
dv/dt = ( gleak*(Vleak-v) + gexc*(Eexc-v) + ginh*(Einh-v)) / (C_m)  : volt
dgexc/dt = -gexc/tau_exc : siemens
dginh/dt = -ginh/tau_inh : siemens
"""

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
    list_for_all_isi = []
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
        
        for element in isi_of_neuron:
            list_for_all_isi.append(element)
        
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
    # print 'isi per neuron: ', isi_per_neuron
    # print 'cv per neuron: ', cv_per_neuron
    average_CV = mean(cv_per_neuron)
    # print 'average_CV', average_CV
    # print cv_per_neuron
    # print average_CV
    if cv_per_neuron != []:
        figname = 'CV_hist' + str(int(10*a)) + str(int(10*b))
        fig = figure()
        title('gext = %s ginh = %s; Coefficient of variance'%(gext, ginh))
        xlabel('CV value')
        ylabel('frequency')
        hist(cv_per_neuron, 50, normed = True)
        fig.savefig(figname + '.png', bbox_inches='tight')
        close(fig)
###############################################################
## Histograms of ISI
###############################################################
        figname = 'hist_ISI_' + str(int(10*a)) + str(int(10*b))
        fig = figure()
        title('gext = %s ginh = %s; ISI'%(gext, ginh))
        xlabel('ISI value')
        ylabel('frequency')
        hist(array(list_for_all_isi), 50, normed = True)

        fig.savefig(figname + '.png', bbox_inches='tight')
        close(fig)
        
        # figname = 'hist2_ISI_' + str(int(10*a)) + str(int(10*b))
        # fig = figure()
        # title('gext = %s ginh = %s; ISI with smaller xrange'%(gext, ginh))
        # xlabel('ISI value')
        # ylabel('frequency')
        # hist(array(list_for_all_isi), 50, range=[0, 100], normed = True)
        # 
        # fig.savefig(figname + '.png', bbox_inches='tight')
    
    return average_CV
   

##############################################################################   

ztemp_cv = []              # Matrix where CV values will be stored in
ztemp_syn = []
yvalues = []            # List where gext values will be stored in
xvalues = []            # List where ginh values will be stored in

kmeansdata_cv = []
kmeansdata_syn = []

gext_lower = 1.5          # Lower bound of gext for loop
gext_upper = 10.5        # Upper bound of gext for loop

ginh_lower = 1            # !!!!!!!!!!!!!!TO CHANGE: number of neurons and epsilon for large simulation !!!!!!!!!!!!!!!!!!!
ginh_upper = 10

stepsize = .5

for i in arange(gext_lower, gext_upper+stepsize, stepsize):
    yvalues.append(i)   # Add gext value to y-axis list

for j in arange(ginh_lower, ginh_upper+stepsize, stepsize):
    xvalues.append(j)   # Add ginh value to x-axis list


for a in arange(gext_lower,gext_upper,stepsize):
    ztemp_cv.append([])    # Add row to matrix
    ztemp_syn.append([])    # Add row to matrix
    
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
        
        
        # # Function to determine rates for excitatory input neurons
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
        
        SM = None
        PRM = None
        run(500*ms, report='stdout')
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
        
        run(record_duration, report='stdout')
        
        absolute_rate_array = (((PRM.rate/khertz)*(N_e+N_i))*(dt/ms))
        biggest_peaks = absolute_rate_array.argsort()[-3:][::-1]
        
        new_frequency_measure = mean(absolute_rate_array[biggest_peaks])
        
        number_of_spikes = int(sum((PRM.rate*(N_e+N_i)/10000))/hertz)
        
        dummy_matrix = zeros((N_e+N_i, len(PRM.rate)))
        nspike_row = numpy.random.randint(N_e+N_i, size=number_of_spikes)
        nspike_column = numpy.random.randint(len(PRM.rate), size=number_of_spikes)
        for i in range(len(nspike_column)):
            dummy_matrix[nspike_row[i]][nspike_column[i]] = 1
            
        dummy_rates = numpy.sum(dummy_matrix, axis=0)
        dummy_biggest_peaks = dummy_rates.argsort()[-3:][::-1]
        dummy_freq_measure = mean(dummy_rates[dummy_biggest_peaks])
        
        synchrony_measure = new_frequency_measure/dummy_freq_measure
        
        figname2 = 'fig' + str(int(10*a)) + str(int(10*b))
        fig2 = figure()
        # exec("%s = figure()"%(figname))

        randomsample = random.sample(xrange(N_e), 25)
        plotlist_t = []
        plotlist_i = []
        for n in range(len(SM.i)):
            if SM.i[n] in randomsample:
                plotlist_t.append(SM.t[n]/ms)
                plotlist_i.append(randomsample.index(SM.i[n]))
        
        subplot(411)
        title('gext = %s ginh = %s; 25 exc neurons; 25 inh neurons; global activity; dummy activity; %s'%(gext, ginh, synchrony_measure))
        ylabel('neuron index')
        plot(plotlist_t, plotlist_i, '.k')
        
        randomsample = random.sample(xrange(N_e, N_e + N_i), 25)
        plotlist_t = []
        plotlist_i = []
        for n in range(len(SM.i)):
            if SM.i[n] in randomsample:
                plotlist_t.append(SM.t[n]/ms)
                plotlist_i.append(randomsample.index(SM.i[n]))
        
        subplot(412)
        ylabel('neuron index')
        plot(plotlist_t, plotlist_i, '.k')
        
        subplot(413)
        ylabel('Frequency')
        xlabel('time (ms)')
        plot(PRM.t/ms, absolute_rate_array)
        
        subplot(414)
        ylabel('Frequency')
        xlabel('time (ms)')
        plot(PRM.t/ms, dummy_rates)

        # 
        # subplot(414)
        # ylabel('Frequency')
        # xlabel('time (ms)')
        # # plot(PRM_e.t/ms, PRM_e.rate*Ne/10000)
        # plot(PRM_i.t/ms, PRM_i.rate)
        
        fig2.savefig(figname2 + '.png', bbox_inches='tight')
        close(fig2)
        # exec("%s = fig"%(figname))

        # stripped_PRMrate = PRM.rate[PRM.rate != 0]
        # stripped_scaled_PRMrate = stripped_PRMrate/sum(stripped_PRMrate)
        # frequency_measure_old = mean(stripped_scaled_PRMrate)
        
        # figname3 = 'StrippedScaledHist' + str(int(10*a)) + str(int(10*b))
        # fig3 = figure()
        # title('gext = %s ginh = %s; distribution of relative frequency. Frequency measure: %s'%(gext, ginh, frequency_measure))
        # hist(stripped_scaled_PRMrate, 50)
        # fig3.savefig(figname3 + '.png', bbox_inches='tight')
        # close(fig3)
        
        av_cv = calculate_cv()
        # print av_cv
        # print a, type(a)
        # print gext_lower, type(gext_lower)
        # print stepsize, type(stepsize)
        # print (a-float(gext_lower))/stepsize
        # print a-float(gext_lower)
        # print int((a-float(gext_lower))/stepsize)
        ztemp_cv[int((a-float(gext_lower))/stepsize)].append(av_cv)
        ztemp_syn[int((a-float(gext_lower))/stepsize)].append(synchrony_measure)
        
        # ISIS = get_isi_from_trains(get_trains_from_spikes(SM.i, SM.t/ms, imax=None), flat=True)
        
        kmeansdata_syn.append(synchrony_measure)
        kmeansdata_cv.append(av_cv)
        
zvalues_cv = ma.array(ztemp_cv, mask=np.isnan(ztemp_cv))
zvalues_syn = ma.array(ztemp_syn, mask=np.isnan(ztemp_syn))
# print zvalues

colorplot_cv = figure()
title('CV values of different gext and ginh values')
xlabel('ginh (nS)')
ylabel('gext (nS)')
colormap = mpl.cm.RdBu
colormap.set_bad('k', 1.)
pcolormesh(array(xvalues), array(yvalues), zvalues_cv, cmap = colormap)
colorbar()

colorplot_cv.savefig('colorplot_cv.png', bbox_inches='tight')


colorplot_syn = figure()
title('Measure of synchrony of different gext and ginh values')
xlabel('ginh (nS)')
ylabel('gext (nS)')
colormap = mpl.cm.RdBu
colormap.set_bad('k', 1.)
pcolormesh(array(xvalues), array(yvalues), zvalues_syn, cmap = colormap)
colorbar()

colorplot_syn.savefig('colorplot_syn.png', bbox_inches='tight')


sio.savemat('Matlab_file.mat', {'zvalues_cv':ztemp_cv, 'zvalues_syn':ztemp_syn, 'xvalues':xvalues, 'yvalues':yvalues})
sio.savemat('Matlab_kmeans.mat', {'kmeansdata_cv':kmeansdata_cv, 'kmeansdata_syn':kmeansdata_syn})

end = time.time()
total_time = end-start
total_time_hours = total_time/3600
print total_time,' seconds'
print total_time_hours, ' hours'

print 'Finished'