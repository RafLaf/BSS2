import matplotlib.pyplot as plt
import numpy as np

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2 import Population
from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

# Welcome to this tutorial of using pyNN for the BrainScaleS-2 neuromorphic
# accelerator.
# We will guide you through all the steps necessary to interact with the
# system and help you explore the capabilities of the on-chip analog neurons
# and synapses.

# To begin with, we configure the logger used during our experiments.
pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
logger = pynn.logger.get("single_neuron_demo")

def plot_membrane_dynamics(population: Population, segment_id=-1):
    """
    Plot the membrane potential of the neuron in a given population view. Only
    population views of size 1 are supported.
    :param population: Population, membrane traces and spikes are plotted for.
    :param segment_id: Index of the neo segment to be plotted. Defaults to
                       -1, encoding the last recorded segment.
    """
    if len(population) != 1:
        raise ValueError("Plotting is supported for populations of size 1.")

    # Experimental results are given in the 'neo' data format, the following
    # lines extract membrane traces as well as spikes and construct a simple
    # figure.
    mem_v = population.get_data("v").segments[segment_id].analogsignals[0].base
    times = mem_v[:, 0]
    membrane = mem_v[:, 1]
    try:
        spikes = population.get_data("spikes").segments[0]

        for spiketime in spikes.spiketrains[0]:
            plt.axvline(spiketime, color="black")
    except IndexError:
        logger.INFO("No spikes found to plot.")
    
    plt.plot(times, membrane, alpha=0.5)
    logger.INFO(f"Mean membrane potential: {np.mean(membrane)}")
    plt.xlabel("Wall clock time [ms]")
    plt.ylabel("Readout (u.a)")
    plt.ylim(0, 1023)  # ADC precision: 10bit -> value range: 0-1023
    return times,membrane





def MADCinv(MADC):
    return (MADC-700)/400*0.6+1

plt.figure()
start=1
stop=500
n=5
listcond=np.linspace(1,1000,n)
listi=np.linspace(start,stop,n)
np.save('conds.npy',listcond)
np.save('cur.npy',listi)
freq=np.zeros((n,n))
thres=400
LL=[]
for i in range(n):
    L=[]
    for j in range(n):
        cur=int(listi[i])
        cond=int(listcond[j])
        pynn.setup()
        # Preparing the neuron to receive synaptic inputs requires the configuration
        # of additional circuits. The additional settings include technical parameters
        # for bringing the circuit to its designed operating point as well as
        # configuration with a direct biological equivalent.
        stimulated_p = pynn.Population(1, pynn.cells.HXNeuron(
        # Leak potential, range: 300-1000
        leak_v_leak=800,
        # Leak conductance, range: 0-1022 increase -> time constant decrease
        leak_i_bias=cond,
        # Threshold potential, range: 0-600
        threshold_v_threshold=400,
        # Reset potential, range: 300-1000
        reset_v_reset=600,
        # Membrane capacitance, range: 0-63  increase->increase time constant 63-->2.4pF
        membrane_capacitance_capacitance=63,
        # Refractory time, range: 0-255
        refractory_period_refractory_time=120,
        # Enable reset on threshold crossing
        threshold_enable=True,
        # Reset conductance, range: 0-1022
        reset_i_bias=1022,
        # Enable strengthening of reset conductance
        reset_enable_multiplication=True,
        # -- Parameters for synaptic inputs -- #
        # Enable synaptic stimulation
        excitatory_input_enable=False,
        inhibitory_input_enable=False,
        # Strength of synaptic inputs
        excitatory_input_i_bias_gm=1022,
        inhibitory_input_i_bias_gm=1022,
        # Synaptic time constants
        excitatory_input_i_bias_tau=200,
        inhibitory_input_i_bias_tau=200,
        # Technical parameters
        excitatory_input_i_drop_input=300,
        inhibitory_input_i_drop_input=300,
        excitatory_input_i_shift_reference=300,
        inhibitory_input_i_shift_reference=300))
        stimulated_p.set(constant_current_enable=True, constant_current_i_offset=cur)
        stimulated_p.record(["v", "spikes"])
        
        '''
        # Create off-chip populations serving as excitatory external spike sources
        exc_spiketimes = [0.01, 0.05, 0.07, 0.08]
        exc_stim_pop = pynn.Population(1, SpikeSourceArray(spike_times=exc_spiketimes))

        # We represent projections as entries in the synapse matrix on the neuromorphic
        # chip. Weights are stored in digital 6bit values (plus sign), the value
        # range for on-chip weights is therefore -63 to 63.
        # With this first projection, we connect the external spike source to the
        # observed on-chip neuron population.
        pynn.Projection(exc_stim_pop, stimulated_p,
                        pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63),
                        receptor_type="excitatory")

        # Create off-chip populations serving as inhibitory external spike sources.
        inh_spiketimes = [0.03]
        inh_stim_pop = pynn.Population(1, SpikeSourceArray(spike_times=inh_spiketimes))

        pynn.Projection(inh_stim_pop, stimulated_p,
                        pynn.AllToAllConnector(),
                        synapse_type=StaticSynapse(weight=63),
                        receptor_type="inhibitory")
        '''
        # You may play around with the parameters in this experiment to achieve
        # different traces. Try to stack multiple PSPs, try to make the neurons spike,
        # try to investigate differences between individual neuron instances,
        # be creative!
        pynn.run(0.1)
        times,data=plot_membrane_dynamics(stimulated_p)
        L.append(data)
        pynn.end()
        if i==0 and j==0:
            np.save('times.npy',times)
    LL.append(L)
    plt.savefig('v'+str(i)+'.pdf')
np.save('LL.npy',np.array(LL))
plt.show()


