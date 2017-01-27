from brian2 import *

# ###########################################
# Defining network model parameters
# ###########################################

simtime = 0.5*second # Simulation time

number = { 'CA3':100, 'I':10, 'CA1':1 }
epsilon = { 'CA3_CA1':0.1,'CA3_I':1.0,'I_CA1':1.0 } # Sparseness of synaptic connections
tau_ampa = 5.0*ms   # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms  # GABAergic synaptic time constant

# ###########################################
# Neuron model
# ###########################################
gl = 10.0*nsiemens   # Leak conductance
el = -60*mV          # Resting potential
er = -80*mV          # Inhibitory reversal potential
vt = -50.*mV         # Spiking threshold
memc = 200.0*pfarad  # Membrane capacitance

eqs_neurons='''
dv/dt= (-gl*(v-el) - (g_ampa*v -g_gaba*(v-er)))/memc : volt (unless refractory)
dg_ampa/dt = -g_ampa/tau_ampa : siemens
dg_gaba/dt = -g_gaba/tau_gaba : siemens '''

# ###########################################
# Interneuron specific 
# ###########################################
delta = 10.*ms

# ###########################################
# Initialize neuron group
# ###########################################

CA3 = SpikeGeneratorGroup(number['CA3'], arange(number['CA3']), 100*ones(number['CA3'])*ms)
I = NeuronGroup(number['I'], model=eqs_neurons, threshold='v > vt', reset='v=el', refractory=1*ms, method='euler')
CA1 = NeuronGroup(number['CA1'], model=eqs_neurons, threshold='v > vt', reset='v=el', refractory=5*ms, method='euler')

# ###########################################
# Connecting the network
# ###########################################

CA3_CA1 = Synapses(CA3, CA1, on_pre='g_ampa += 0.3*nS')
CA3_CA1.connect(p=epsilon['CA3_CA1'])

CA3_I = Synapses(CA3, I, on_pre='g_ampa += 0.3*nS')
CA3_I.connect(p=epsilon['CA3_I'])

I_CA1 = Synapses(I, CA1, on_pre='g_gaba += 0.3*nS',delay=delta)
I_CA1.connect(p=epsilon['I_CA1'])

# ###########################################
# Setting up monitors
# ###########################################

sm = SpikeMonitor(CA3)
sm_i = SpikeMonitor(I)
trace = StateMonitor(CA1, 'v', record=True)

# ###########################################
# Run without plasticity
# ###########################################
run(simtime)

# ###########################################
# Make plots
# ###########################################

i, t = sm.it
subplot(111)
plot(t/ms, i, 'k-', ms=0.25)
title("Before")
xlabel("time (ms)")
yticks([])
xlim(0*1e3, 2*1e3)
show()

i, t = sm_i.it
subplot(111)
plot(t/ms, i, 'r-', ms=0.2, markersize='100')
title("Before")
xlabel("time (ms)")
yticks([])
xlim(0*1e3, 2*1e3)
show()


v = trace.v[0].T
t = trace.t
subplot(111)
plot(t/ms, v, 'k', ms=0.1)
ylim((-0.065,-0.058) )
xlabel("time (ms)")
title("Voltage")
show()
