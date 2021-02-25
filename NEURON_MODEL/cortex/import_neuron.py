from neuron import h
import numpy as np
import matplotlib.pyplot as plt


#h.load_file(filename+'mosinit.hoc')
#h.load_file(filename+'stdrun.hoc')
#h.load_file(filename+"nrngui.hoc")




h.load_file("stdrun.hoc")
h.load_file(1,"demo_PY_RSnoGUI_Amp0.oc")
def response(x):
    if x.shape!=(10000,):
        print('your input curent must be an array of shape (1000,) ')
    print(h.PY[0])
    stim=h.IClamp(0.5,sec=h.PY[0].soma[0])
    stim.delay=0                              #necessary to choose the Iclamp see documentation.
    stim.dur=1e9
    ramp=h.Vector(x)
    ramp.play(stim._ref_amp,0.1)
    vm=h.Vector()
    vm.record(h.PY[0].soma[0](0.5)._ref_v)
    h.init()
    print(h.t)
    h.run()
    print(h.t)
    return np.array(vm)


stepresponse=[]
steps=np.linspace(-1,5,50)
for step in steps:
    x=np.zeros(10000)
    x[2000:]=step
    stepresponse.append(response(x))
stepresponse=np.array(stepresponse)
np.save('/home/raphael/Documents/BSS2/recodingsinNeuron/importneuron/steps.npy',steps)
np.save('/home/raphael/Documents/BSS2/recodingsinNeuron/importneuron/stepreponse.npy',stepresponse)

rampresponse=[]
ramps=np.linspace(0,10,50)
for ramp in ramps:
    x=np.linspace(0,ramp,10000)
    rampresponse.append(response(x))
rampresponse=np.array(rampresponse)
np.save('/home/raphael/Documents/BSS2/recodingsinNeuron/importneuron/ramps.npy',ramps)
np.save('/home/raphael/Documents/BSS2/recodingsinNeuron/importneuron/rampreponse.npy',rampresponse)


