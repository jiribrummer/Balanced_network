from brian2 import *
print 'start'
# # # 
# # # # # Learning rule
# # # # v = -69 
# # # # F = -1 / (1 + exp(-(v+55 * mV)/(4 * mV))) + .5 * (2 * mV) * log(1 + exp((v+(52 * mV )/(2 * mV))))
# # # # F = (-1 / (1 + exp(-((v+55)/(4))))) + .5 * 2 * log10(1 + exp((v+52)/(2)))
# # # 
# # # F = 0
# # # v = -69
# # # v0 = -55
# # # v1 = -52
# # # s0 = 4
# # # s1 = 2
# # # a = 0.5
# # # 
# # # 
# # # 
# # # def calc(v, v0, v1, s0, s1, a):
# # #     F = -1 * ((1)/(1 + (exp((-((v-v0)/s0)))))) + (a *s1 * log10(1 + exp((v-v1)/s1)))
# # #     return F
# # # 
# # # for v in [-50, -51, -60, -69, -70]:
# # #     print v
# # #     print calc(v, v0, v1, s0, s1, a)


N = 3
neuron_spacing = 50*umetre
width = N/4.0*neuron_spacing

# Neuron has one variable x, its position
G = NeuronGroup(N, 'x : metre')
G.x = 'i*neuron_spacing'

# All synapses are connected (excluding self-connections)
S = Synapses(G, G, 'w : 1')
S.connect('i!=j')
# Weight varies with distance
S.w = 'exp(-(x_pre-x_post)**2/(2*width**2))'

scatter(G.x[S.i]/um, G.x[S.j]/um, S.w*20)
xlabel('Source neuron position (um)')
ylabel('Target neuron position (um)')

show()