import pyNN.neuron as sim  # can of course replace `nest` with `neuron`, `brian`, etc.
import matplotlib.pyplot as plt
from quantities import nA
from pyNN.random import RandomDistribution, NumpyRNG

refractory_period=RandomDistribution('uniform', [2.0, 3.0], rng=NumpyRNG(seed=4242))


sim.setup()
pyr_parameters= {'cm': 0.25, 'tau_m': 20.0, 'v_rest': -60, 'v_thresh': -50, 'tau_refrac': refractory_period, 'v_reset': -60, 'v_spike': -50.0, 'a': 1.0, 'b': 0.005, 'tau_w': 600, 'delta_T': 2.5,  'tau_syn_E': 5.0, 'e_rev_E': 0.0, 'tau_syn_I': 10.0, 'e_rev_I': -80 }

pyrcell = sim.Population(1, sim.EIF_cond_exp_isfa_ista(**pyr_parameters))

step_current = sim.DCSource(start=20.0, stop=80.0)
step_current.inject_into(pyrcell)

pyrcell.record('v')
print(pyrcell.celltype.recordable)

for amp in (-0.2, -0.1, 0.0, 0.1, 0.2,0.3,0.4,0.5):
    step_current.amplitude = amp
    sim.run(150.0)
    sim.reset(annotations={"amplitude": amp * nA})

data = pyrcell.get_data()

sim.end()

for segment in data.segments:
    vm = segment.analogsignals[0]
    plt.plot(vm.times, vm,
             label=str(segment.annotations["amplitude"]))
plt.legend(loc="upper left")
plt.xlabel("Time (%s)" % vm.times.units._dimensionality)
plt.ylabel("Membrane potential (%s)" % vm.units._dimensionality)

plt.show()