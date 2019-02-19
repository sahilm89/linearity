########################################################################
# This example demonstrates divisive normalization
# Copyright (C) Upinder S. Bhalla NCBS 2018
# Released under the terms of the GNU Public License V3.
########################################################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
import Tkinter, tkFileDialog, tkMessageBox
import scipy.stats as ss
import numpy as np
import warnings
import moose
import rdesigneur as rd
import sys
sys.path.append('../')
from Linearity import Neuron

lines = []
tplot = ""
axes = []
sliders = []

RM = 1.0
RA = 1.0
CM = 0.01
dia = 10e-6
runtime = 0.1
elecDt = 50e-6
elecPlotDt = 50e-6 
sliderMode = "Gbar"
gluGbar = 0.001
IE_ratio = 1.0
IE_ratio_arr = []
dynamicDelay = 6.37 
dynamicDelay_toggle = 1
gabaGbar = IE_ratio*gluGbar
K_A_Gbar_init = 0.  #K_A_Gbar why did Upi use 2?
K_A_Gbar = K_A_Gbar_init 
gluOnset = 20.0e-3
gabaOnset = 22.0e-3
minDelay = gluOnset*1e3 + 2
max_synapticDelay = gluOnset*1e3 +15
max_exc_cond = 0.7
inputFreq = 100.0
inputDuration = 0.01
printOutput = True
makeMovie = True
frameNum = 0
fname = "movie/randomInput/frame"
ttext = ""
maxVolt = 20.

exc_vec = np.zeros(int(runtime/elecDt)) 
inh_vec = np.zeros(int(runtime/elecDt))

spikingDistrib = []
K_A_distrib = [['K_A', 'soma', 'Gbar', str(K_A_Gbar) ]]

rec = []
sdn_x = []
sdn_y = []
last_pk = [0., 0.]
gabaonset_list = []
gluGbar_list = []
peaktime_list = []
max_g_exc = []


def simpleaxis(axes, every=False, outward=False):
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if (outward):
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))
        if every:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title('')

def get_dir():
    root = Tkinter.Tk()
    root.withdraw()
    directory = tkFileDialog.askdirectory(initialdir='.')
    return directory

def findOnsetTime(trial, expType, step=0.5, slide = 0.05, minOnset = 2., maxOnset = 50., initpValTolerance=1.0, pValMinTolerance = 0.1):
    maxIndex = int(trial.F_sample*maxOnset*1e-3)
    if expType == 1:
        maxOnsetIndex = np.argmax(-trial.interestWindow[:maxIndex])
    elif expType == 2:
        maxOnsetIndex = np.argmax(trial.interestWindow[:maxIndex])
    else:
        maxOnsetIndex = np.argmax(trial.interestWindow[:maxIndex])
    
    window_size = len(trial.interestWindow)
    step_size = int(trial.F_sample*step*1e-3)
    
    overlap =  int(trial.F_sample*slide*1e-3)
    
    index_right = maxOnsetIndex
    index_left = index_right - step_size
    minOnsetIndex = int(trial.F_sample*minOnset*1e-3)
    
    baseMean = np.mean(trial.interestWindow[:minOnsetIndex])
    factor = 5
    thresholdGradient = 0.01
    pValTolerance = initpValTolerance

    l_window = trial.interestWindow[:minOnsetIndex]
    while (index_left>minOnset):
        r_window = trial.interestWindow[index_left:index_right] #, trial.baselineWindow #trial.interestWindow[index_left - step_size:index_left]
        stat, pVal = ss.ks_2samp(r_window, l_window)
        if pVal>pValTolerance:
            return float(index_right)/trial.F_sample
        else:
            index_left-=overlap
            index_right-=overlap
            if index_left<=minOnsetIndex:
                pValTolerance/=2
                if pValTolerance<pValMinTolerance:
#                             print ("Returning Nan")
                            return np.nan
                else:
                    index_right = maxOnsetIndex
                    index_left = maxOnsetIndex - step_size

def get_EI_pairs(n):

    obs_exc, obs_inh = {}, {}
    exc_error, inh_error = {}, {}
    del_exc, del_inh = {}, {}
    tau_exc_dict, tau_inh_dict, delta_exc_dict, delta_inh_dict, g_exc_dict, g_inh_dict = {}, {}, {}, {}, {}, {}
    ttp_exc, ttp_inh = {}, {}
    amplitude = {}
    sqrs = []
    coord_sqrs = {}

    EI_pair = {}
    E, I = {}, {}
    E_onsetDelay, I_onsetDelay = {}, {}
    for expType,expt in n:
        for sqr in expt:
            #if sqr == 1 or sqr == 2:
                if (expType == 1):
                    sqrs.append(sqr)
                    for coord in expt[sqr].coordwise:

                        E[coord] = [] 
                        E_onsetDelay[coord] = []

                        delays = []
                        for trial in expt[sqr].coordwise[coord].trials:
                            E[coord].append(trial.interestWindow)
                            onset = findOnsetTime(trial, expType)
                            if onset:
                                E_onsetDelay[coord].append(onset)
                        #obs_exc[coord] = np.average(trajectory,axis=0)*nanosiemens/-70
                        #del_exc[coord] = np.nanmean(delays)
                        #t_arr = np.linspace(0,100,len(obs_exc[coord]))
                        #ttp_exc[coord] = t_arr[np.argmax(obs_exc[coord])]

                elif (expType == 2):
                    for coord in expt[sqr].coordwise:

                        I[coord] = [] 
                        I_onsetDelay[coord] = []
                        delays = []
                        for trial in expt[sqr].coordwise[coord].trials:
                            I[coord].append(trial.interestWindow)
                            onset = findOnsetTime(trial, expType)
                            if onset:
                                I_onsetDelay[coord].append(onset)
                        #obs_inh[coord] = np.average(trajectory,axis=0)*nanosiemens/70
                        #del_inh[coord] = np.nanmean(delays)
                        #t_arr = np.linspace(0,100,len(obs_inh[coord]))
                        #ttp_inh[coord] = t_arr[np.argmax(obs_inh[coord])]

    for coord in set(E.keys()).intersection(set(I.keys())):
        EI_pair[coord] = zipper(E[coord], I[coord])
        EI_onset[coord] = zipper(E_onsetDelay[coord],I_onsetDelay[coord])
    return EI_pair, EI_onset

def zipper (list1, list2):
    ''' Zips together a list of lists and returns tuples '''
    paired_tuple = []
    for l1 in list1:
        for l2 in list2:
            paired_tuple.append((l1,l2))
    return tuple(paired_tuple)

class fileDialog():

    def __init__( self, ax ):
        self.duration = 1
        self.ax = ax

    def click( self, event ):
        dirname = get_dir()
        cellIndex = dirname.split('/')[-1]
        filename = dirname + '/plots/' + cellIndex + '.pkl'
        print(filename)
        n = Neuron.load(filename)
        print("Loaded {} {}".format(n.date, n.index))
        EI_pair, EI_onset = get_EI_pairs(n)
        for coord in EI_pair.keys():
            for j, synTuple in enumerate(EI_pair[coord]):
                set_EmpSynVec(synTuple)
                set_EmpOnset(EI_onset[coord][j])

def set_EmpSynVec(synTuple):
    global exc_vec, inh_vec
    exc_vec, inh_vec = synTuple
    updateDisplay()

def set_EmpOnset(EI_onset):
    global gluOnset, gabaOnset 
    gluOnset, gabaOnset = EI_onset 
    updateDisplay()


def setGluGbar( val ):
    global gluGbar
    gluGbar = val
    setGabaGbar(gluGbar*IE_ratio, update=False)
    if dynamicDelay_toggle:
        setDynamicDelay(dynamicDelay)
    else:
        updateDisplay()

def setStaticDelay(val):
    global gluOnset 
    setGabaOnset(val+gluOnset*1e3)

def setDynamicDelay( val):
    global dynamicDelay
    dynamicDelay = val
    #print("Delay debugging values", minDelay, max_synapticDelay,dynamicDelay, np.exp(-dynamicDelay*gluGbar))
    newOnset = minDelay + (max_synapticDelay - minDelay)*np.around(np.exp(-dynamicDelay*gluGbar),decimals=3)
    print("New onset is ", newOnset)
    setGabaOnset( newOnset)

def setIE_ratio( val ):
    global IE_ratio 
    IE_ratio = val
    setGabaGbar(gluGbar*IE_ratio)

def setGabaGbar( val, update=True ):
    global gabaGbar
    gabaGbar = val
    if update:
        updateDisplay()

def setK_A_Gbar( val ):
    global K_A_Gbar
    K_A_Gbar = val
    updateDisplay()

def setK_A_Gbar_to_zero(val):
    global K_A_Gbar
    K_A_Gbar= 0.
    updateDisplay()

def setGabaOnset( val ):
    global gabaOnset
    gabaOnset = val/1000.0
    updateDisplay()

def setRM( val ):
    global RM
    RM = val
    updateDisplay()

def setCM( val ):
    global CM
    CM = val
    updateDisplay()

def makeModel():
    cd = [
            #['glu', 'soma', 'Gbar', str(gluGbar)],
            #['GABA', 'soma', 'Gbar', str(gabaGbar)],
            ['K_A', 'soma', 'Gbar', str(K_A_Gbar) ],
            ['exc', 'soma', 'Gbar', '0.'],
            ['inh', 'soma', 'Gbar', '0.']
    ]
    cd.extend( spikingDistrib )
     
    rdes = rd.rdesigneur(
        elecDt = elecDt,
        elecPlotDt = elecPlotDt,
        stealCellFromLibrary = True,
        verbose = False,
        chanProto = [
            #['make_glu()', 'glu'],['make_GABA()', 'GABA'],
            #['make_K_A()','K_A'],
            ['make_Na()', 'Na'],['make_K_DR()', 'K_DR'],
            ['make_EmpExc()', 'exc'], ['make_EmpInh()', 'inh']
        ],
        cellProto = [['somaProto', 'cellBase', dia, dia]],
        passiveDistrib = [[ '#', 'RM', str(RM), 'CM', str(CM), 'RA', str(RA) ]],
        chanDistrib = cd,
        stimList = [
            #['soma', '1','glu', 'periodicsyn', '{}*(t>{:.6f} && t<{:.6f})'.format( inputFreq, gluOnset, gluOnset + inputDuration) ],
            #['soma', '1','GABA', 'periodicsyn', '{}*(t>{:.6f} && t<{:.6f})'.format( inputFreq, gabaOnset, gabaOnset + inputDuration) ],
        ],
        plotList = [['soma', '1','.', 'Vm']] #, ['soma', '1','exc', 'Ik'], ['soma', '1','inh', 'Ik'], ['soma', '1','exc', 'Gk'], ['soma', '1','inh', 'Gk']],
    )
    #moose.element( '/library/GABA' ).Ek = -0.07
    
    rdes.buildModel()

    exc = moose.element('/library/exc')
    inh = moose.element('/library/inh')

    moose.connect(exc, 'channel', '/model/elec/soma', 'channel')
    moose.connect(inh, 'channel', '/model/elec/soma', 'channel')

    excData = moose.element( '/library/exc/data')
    inhData = moose.element( '/library/inh/data')

    excData.vector = exc_vec*-1e-3
    inhData.vector = inh_vec*1e-3

    tab_exc = moose.Table('/model/graphs/plot1')
    tab_inh = moose.Table('/model//graphs/plot2')

    moose.connect(tab_exc, 'requestOut', exc, 'getGk')
    moose.connect(tab_inh, 'requestOut', inh, 'getGk')

    moose.le('/model/graphs')

def makeModelWithoutInhibition():
    cd = [
            #['glu', 'soma', 'Gbar', str(gluGbar)],
            #['GABA', 'soma', 'Gbar', str(gabaGbar)],
            ['K_A', 'soma', 'Gbar', str(K_A_Gbar) ],
            ['exc', 'soma', 'Gbar', '0.'],
            ['inh', 'soma', 'Gbar', '0.']
    ]
    cd.extend( spikingDistrib )
     
    rdes = rd.rdesigneur(
        elecPlotDt = elecPlotDt,
        stealCellFromLibrary = True,
        verbose = False,
        chanProto = [
            #['make_glu()', 'glu'],['make_GABA()', 'GABA'],
            #['make_K_A()','K_A'],
            ['make_Na()', 'Na'],['make_K_DR()', 'K_DR'],
            ['make_EmpExc()', 'exc'], ['make_EmpInh()', 'inh']
        ],
        cellProto = [['somaProto', 'cellBase', dia, dia]],
        passiveDistrib = [[ '#', 'RM', str(RM), 'CM', str(CM), 'RA', str(RA) ]],
        chanDistrib = cd,
        stimList = [
            #['soma', '1','glu', 'periodicsyn', '{}*(t>{:.6f} && t<{:.6f})'.format( inputFreq, gluOnset, gluOnset + inputDuration) ],
            #['soma', '1','GABA', 'periodicsyn', '{}*(t>{:.6f} && t<{:.6f})'.format( inputFreq, gabaOnset, gabaOnset + inputDuration) ],
        ],
        plotList = [['soma', '1','.', 'Vm']] #, ['soma', '1','exc', 'Ik'], ['soma', '1','inh', 'Ik'], ['soma', '1','exc', 'Gk'], ['soma', '1','inh', 'Gk']],
    )
    #moose.element( '/library/GABA' ).Ek = -0.07
    
    rdes.buildModel()

    exc = moose.element('/library/exc')
    inh = moose.element('/library/inh')

    moose.connect(exc, 'channel', '/model/elec/soma', 'channel')
    moose.connect(inh, 'channel', '/model/elec/soma', 'channel')

    excData = moose.element( '/library/exc/data')
    inhData = moose.element( '/library/inh/data')
    #excData.vector = np.random.normal(0,1,10000)*1e-10
    excData.vector = exc_vec*-1e-3 
    inhData.vector = np.zeros(len(exc_vec))*1e-3

    tab_exc = moose.Table('/model/graphs/plot1')
    tab_inh = moose.Table('/model//graphs/plot2')

    moose.connect(tab_exc, 'requestOut', exc, 'getGk')
    moose.connect(tab_inh, 'requestOut', inh, 'getGk')

def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    makeDisplay()
    quit()

class stimToggle():
    def __init__( self, toggle, ax ):
        self.duration = 1
        self.toggle = toggle
        self.ax = ax

    def click( self, event ):
        global spikingDistrib
        if self.duration < 0.5:
            self.duration = 1.0
            self.toggle.label.set_text( "Spiking off" )
            self.toggle.color = "yellow"
            self.toggle.hovercolor = "yellow"
            spikingDistrib = []
        else:
            self.duration = 0.001
            self.toggle.label.set_text( "Spiking on" )
            self.toggle.color = "orange"
            self.toggle.hovercolor = "orange"
            spikingDistrib = [['Na', 'soma', 'Gbar', '200' ],['K_DR', 'soma', 'Gbar', '250' ]]
        updateDisplay()

class kaToggle():
    def __init__( self, toggle, ax, sliderIndex):
        self.duration = 1
        self.toggle = toggle
        self.ax = ax
        self.slider_index = sliderIndex

    def click( self, event ):
        global K_A_distrib
        global K_A_Gbar
        global sliders 
        if self.duration < 0.5:
            self.duration = 1.0
            self.toggle.label.set_text( "KA off" )
            self.toggle.color = "yellow"
            self.toggle.hovercolor = "yellow"
            ax_slider = sliders[self.slider_index].ax
            ax_slider.clear() 
            K_A_Gbar=0.
            sliders[self.slider_index].__init__(ax_slider, "Zero KA", 0, 0, K_A_Gbar, dragging=0)

        else:
            self.duration = 0.001
            self.toggle.label.set_text( "KA on" )
            self.toggle.color = "orange"
            self.toggle.hovercolor = "orange"
            ax_slider = sliders[self.slider_index].ax
            ax_slider.clear() 
            K_A_Gbar = K_A_Gbar_init
            sliders[self.slider_index].__init__(ax_slider, "K_A_Gbar (Mho/m^2)", 1, 100, K_A_Gbar, dragging=1)

        sliders[self.slider_index].on_changed( setK_A_Gbar )
        updateDisplay()

class dynamicDelayToggle():
    def __init__( self, toggle, ax, dynDel_index ):
        self.duration = 1
        self.toggle = toggle
        self.ax = ax
        self.dynDel_index = dynDel_index
        dynamicDelay_toggle=0

    def click( self, event ):
        global gabaOnset
        global dynamicDelay_toggle
        if self.duration < 0.5:
            self.duration = 1.0
            self.toggle.label.set_text( "Static Delay" )
            self.toggle.color = "yellow"
            self.toggle.hovercolor = "yellow"

            ax_slider = sliders[self.dynDel_index].ax
            ax_slider.clear() 
            sliders[self.dynDel_index].__init__(ax_slider, "Static Inh. Delay", minDelay-gluOnset*1e3, max_synapticDelay-gluOnset*1e3, (gabaOnset-gluOnset)*1000)
            sliders[self.dynDel_index].on_changed( setStaticDelay )
            dynamicDelay_toggle = 0
            
        else:
            self.duration = 0.001
            self.toggle.label.set_text( "Dynamic Delay" )
            self.toggle.color = "orange"
            self.toggle.hovercolor = "orange"
            ax_slider = sliders[self.dynDel_index].ax
            ax_slider.clear() 
            sliders[self.dynDel_index].__init__(ax_slider, "Dynamic Inh. Delay", 1, 10.0, 6.37)
            sliders[self.dynDel_index].on_changed( setDynamicDelay )
            dynamicDelay_toggle = 1

def printSomaVm():
    print("This is somaVm" )

def updateDisplay():
    global frameNum
    global sdn_x
    global sdn_y
    global IE_ratio_arr
    global last_pk
    global gabaonset_list
    global gluGbar_list
    global peaktime_list
    global K_A_Gbar
    global max_g_exc
    global e_g_peak
    global exc_vec
    global inh_vec

    #print (K_A_Gbar)
    makeModel()
    moose.reinit()
    moose.start( runtime )
    tabvec = moose.element( '/model/graphs/plot0' ).vector
    moose.le('/model/graphs/')
    tabvec_filtered = tabvec[int(gluOnset/elecPlotDt):]
    #print "############## len tabvec = ", len(tabvec)
    print(int(gluOnset/elecPlotDt))
    maxval = max(tabvec_filtered)
    print(maxval)
    imaxval = int(gluOnset/elecPlotDt) + list(tabvec_filtered).index( maxval )
    maxt = imaxval * elecPlotDt * 1000
    pk = (maxval - min( tabvec[:imaxval+1] )) * 1000
    ttext.set_text( "Peak amp.= {:.1f} mV \nPeak time = {:.1f} ms".format( pk, maxt  ) )
    tplot.set_ydata( tabvec * 1000 )

    norm = matplotlib.colors.Normalize(vmin=0.,vmax=7)
    tplot.set_color(plt.cm.plasma(norm(IE_ratio)))
    last_pk[1] = pk 
    
    #exc_i = moose.element( '/model/graphs/plot1' ).vector
    #inh_i = moose.element( '/model/graphs/plot2' ).vector
    #e_plot.set_ydata(exc_i*1e12)
    #i_plot.set_ydata(inh_i*1e12)

    exc_g = moose.element( '/model/graphs/plot1' ).vector
    inh_g = moose.element( '/model/graphs/plot2' ).vector
    e_g_plot.set_ydata(exc_g*1e9)
    i_g_plot.set_ydata(inh_g*1e9)

    print("Sent in/recovered = {}".format(np.array(exc_vec)/np.array(exc_g[1:])))

    print(gabaonset_list, max_g_exc)
    del_exc_scat.set_xdata(gabaonset_list)
    del_exc_scat.set_ydata(max_g_exc)
    #del_exc_scat.set_array(np.array(IE_ratio_arr))

    max_g_exc.append(max(exc_g*1e9))

    i_g_onset.set_xdata([gabaOnset*1e3])
    #i_g_onset.set_color(plt.cm.plasma((norm(IE_ratio))))

    e_g_peak.set_ydata([max(exc_g*1e9)])
    #e_g_peak.set_color(plt.cm.plasma((norm(IE_ratio))))
    ion_text.set_x(gabaOnset*1e3 + 3)
    ep_text.set_y(max(exc_g*1e9) + 0.05)

    if printOutput:
        print( "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format( maxval*1000, pk,maxt, gluGbar, gabaGbar, K_A_Gbar, gabaOnset*1000, RM, CM ) )
   
    moose.delete( '/model' )
    moose.delete( '/library' )

    makeModelWithoutInhibition()
    moose.reinit()
    moose.start( runtime )
    tabvec = moose.element( '/model/graphs/plot0' ).vector
    #print "############## len tabvec = ", len(tabvec)
    maxval = max(tabvec)
    imaxval = list(tabvec).index( maxval )
    maxt_exc = imaxval * elecPlotDt * 1000
    pk = (maxval - min( tabvec[:imaxval+1] )) * 1000
    tplot_noinh.set_ydata( tabvec * 1000 )

    last_pk[0] = pk 
    
    moose.delete( '/model' )
    moose.delete( '/library' )

    last_point.set_data(last_pk)
    last_point.set_data(last_pk)
    sdn_plot.set_offsets(np.array(zip(sdn_x, sdn_y)))
    sdn_plot.set_array(np.array(IE_ratio_arr))
    print("DIMS are", np.shape(np.array(zip(sdn_x, sdn_y))))
    #sdn_plot.set_xdata(sdn_x)
    #sdn_plot.set_ydata(sdn_y)
    #sdn_plot.set_color(IE_ratio_arr)

    sdn_x.append(last_pk[0])
    sdn_y.append(last_pk[1])
    IE_ratio_arr.append(IE_ratio)
 
    #dynDel_plot.set_xdata(gluGbar_list)
    #dynDel_last_point.set_xdata(gluGbar)
 
    #dynDel_plot.set_ydata(gabaonset_list)
    #dynDel_last_point.set_ydata( (gabaOnset- gluOnset)*1000 )
    gabaonset_list.append(gabaOnset*1000)

    #peaktime_plot.set_xdata(gluGbar_list)
    #peaktime_last_point.set_xdata(gluGbar)

    #peaktime_plot.set_ydata(peaktime_list)
    #peaktime_last_point.set_ydata( maxt )
    #peaktime_list.append(maxt)
    #print (maxt)

    gluGbar_list.append(gluGbar)
    
    print(frameNum)
    if makeMovie:
        plt.savefig( "{}_{:04d}.png".format(fname, frameNum) )
        frameNum += 1
 
def doQuit( event ):
    quit()

def makeDisplay():
    global lines
    global tplot
    global tplot_noinh
    global sdn_plot
    global last_point
    global dynDel_plot
    global dynDel_last_point
    global axes
    global sliders
    global ttext
    global ax1
    global ax2
    global ax3
    global ax3_peaktime
    global peaktime_plot
    global peaktime_last_point
    global e_plot
    global i_plot
    global e_g_plot
    global i_g_plot
    global e_g_peak
    global i_g_onset
    global del_exc_scat
    global ep_text
    global ion_text

    #img = mpimg.imread( 'EI_input.png' )
    img = mpimg.imread( 'Image_simulation_3.png' )
    fig = plt.figure( figsize=(10,12) )
    fig.text(0.03,0.9, "a", fontsize=14, fontweight='bold')
    fig.text(0.48,0.9, "b", fontsize=14, fontweight='bold')
    fig.text(0.03,0.61, "c", fontsize=14, fontweight='bold')
    fig.text(0.48,0.61, "d", fontsize=14, fontweight='bold')
    gridspec.GridSpec(3,2)
    cmap = plt.get_cmap('plasma')

    #png = fig.add_subplot(311)
    png = plt.subplot2grid((3,2), (0,0), colspan=1, rowspan=1)
    imgplot = plt.imshow( img )
    plt.axis('off')

    t = np.arange( 0.0, runtime + elecPlotDt / 2.0, elecPlotDt ) * 1000 #ms

    ei_g_ax = plt.subplot2grid((3,2), (0,1), colspan=1, rowspan=1)
    simpleaxis(ei_g_ax)
    plt.ylabel( '$g_{syn}$ (nS)', fontsize=12 )
    #plt.xlabel( 'Time (ms)' )
    plt.ylim(0,1.5)
    plt.title( "Synaptic Conductances" )

    #print "############## len t = ", len(t)
    e_g_plot, = ei_g_ax.plot( t, np.zeros(len(t)), '-', color='blue')
    i_g_plot, = ei_g_ax.plot( t, np.zeros(len(t)), '-', color='orange')
    #e_g_peak, = ei_g_ax.vlines( 0., 0.,1.5, linestyle='--', alpha=0.3,color='blue')
    i_g_onset = ei_g_ax.axvline( gabaOnset*1e3, linestyle='--', alpha=0.3,color='orange')
    e_g_peak = ei_g_ax.axhline( 0., linestyle='--', alpha=0.3,color='blue')

    ion_text = plt.text(gabaOnset*1e3+1, 1.4, "Inh. Onset",color='orange', fontweight='bold') 
    ep_text = plt.text( max(t), 0, "Exc. peak" ,color='blue', fontweight='bold')
    del_exc_scat, = ei_g_ax.plot([], [], 'o', color='orange', markersize=2)

    #t = np.arange( 0.0, runtime + elecPlotDt / 2.0, elecPlotDt ) * 1000 #ms

    #ei_ax = plt.subplot2grid((3,2), (0,2), colspan=1, rowspan=1)
    #simpleaxis(ei_ax)
    #plt.ylabel( '$I$ (pA)' )
    #plt.xlabel( 'Time (ms)' )
    #plt.ylim(-20,20)
    #plt.title( "Synaptic Currents" )

    ##print "############## len t = ", len(t)
    #e_plot, = ei_ax.plot( t, np.zeros(len(t)), '-', color='blue')
    #i_plot, = ei_ax.plot( t, np.zeros(len(t)), '-', color='orange')

    ax1 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
    simpleaxis(ax1)
    #ax1 = fig.add_subplot(312)
    ttext = plt.text( 0, -35, "Peak amp.= 0\n Peak time = 0)", alpha = 0.9 )
    plt.ylabel( '$V_m$ (mV)' , fontsize=12 )
    plt.ylim( -80, -25 )
    plt.xlabel( 'Time (ms)' , fontsize=12 )
    #plt.title( "Soma" )
    t = np.arange( 0.0, runtime + elecPlotDt / 2.0, elecPlotDt ) * 1000 #ms
    #print "############## len t = ", len(t)

    tplot, = ax1.plot( t, np.zeros(len(t)), '-' )
    tplot_noinh, = ax1.plot( t, np.zeros(len(t)), '--',color='gray' )

    ax2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
    simpleaxis(ax2)
    plt.ylabel( 'Observed $V_m$ (mV)' , fontsize=12 )
    #plt.ylim( -80, -30 )
    plt.xlabel( 'Expected $V_m$ (mV)' , fontsize=12 )
    #plt.title( "SDN" )
    ax2.plot([0,1], [0,1], '--', transform=ax2.transAxes)
    ax2.set_xlim( 0, maxVolt )
    ax2.set_ylim( 0, maxVolt )
    #sdn_plot, = ax2.plot(sdn_x, sdn_y, 'o', markersize=6, markerfacecolor=IE_ratio, cmap=cmap)
    sdn_plot = ax2.scatter([], [], s=12, c=[], cmap=cmap, vmin=0., vmax=7.)
    last_point, = ax2.plot(last_pk[0], last_pk[1], 'o', markersize=7, markerfacecolor='k')
    sdn_cbar = plt.colorbar(sdn_plot, ax=ax2, shrink=0.8)
    sdn_cbar.ax.set_title(" I/E")

    #ax3 = plt.subplot2grid((3,2), (1,2), colspan=1, rowspan=1)
    #simpleaxis(ax3)
    #plt.ylabel( 'Onset delay $\delta_{inh}$ (ms)' )
    ##plt.ylim( -80, -30 )
    #plt.xlabel( 'Exc $g_{max}$ (Mho/m^2)' )
    #ax3.set_xlim( 0., max_exc_cond )
    #ax3.set_ylim( 0.,  max_synapticDelay-gluOnset*1000)
    #dynDel_plot, = ax3.plot(gluGbar_list, gabaonset_list, 'o', markersize=4, markerfacecolor='gray',markeredgecolor='gray')
    #dynDel_last_point, = ax3.plot([], [], 'o', markersize=5, markerfacecolor='k')
    #ax3.hlines(y=minDelay-gluOnset*1000, xmin=0, xmax= max_exc_cond, linestyle='--')
    #ax3_peaktime = ax3.twinx()
    #ax3_peaktime.set_xlim( 0., max_exc_cond )
    #ax3_peaktime.set_ylim( 20,  40.)
    #ax3_peaktime.set_ylabel( 'Peak time (ms)' )

    #peaktime_plot, = ax3_peaktime.plot(gluGbar_list, peaktime_list, '^', markersize=5, markerfacecolor='green', markeredgecolor='green')
    #peaktime_last_point, = ax3_peaktime.plot([], [], '^', markersize=5, markerfacecolor='k')

    #ax3.spines['left'].set_color('gray')
    #ax3.spines['left'].set_linewidth('3')
    #ax3_peaktime.spines['right'].set_color('green')
    #ax3_peaktime.spines['right'].set_linewidth('3')
    #ax3_peaktime.spines['top'].set_visible(False)

    #ax = fig.add_subplot(313)
    ax = plt.subplot2grid((3,3), (2,0), colspan=1, rowspan=1)
    plt.axis('off')
    axcolor = 'palegreen'
    axStim = plt.axes( [0.02,0.005, 0.20,0.04], facecolor='green' )
    #axKA = plt.axes( [0.14,0.005, 0.10,0.03], facecolor='green' )
    axLoad = plt.axes( [0.24,0.005, 0.20,0.04], facecolor='green' )
    axReset = plt.axes( [0.46,0.005, 0.20,0.04], facecolor='blue' )
    axQuit = plt.axes( [0.68,0.005, 0.30,0.04], facecolor='blue' )

    for x in np.arange( 0.11, 0.26, 0.06 ):
        axes.append( plt.axes( [0.25, x, 0.65, 0.04], facecolor=axcolor ) )

    sliders.append( Slider( axes[2], "gluGbar (Mho/m^2)", 0.001, max_exc_cond, valinit = gluGbar))

    sliders[-1].on_changed( setGluGbar )
    sliders.append( Slider( axes[0], "I/E ratio", 0.001, 6., valinit = IE_ratio) )
    sliders[-1].on_changed( setIE_ratio )
    #sliders[-1].on_changed( setK_A_Gbar )

    #sliders.append( Slider( axes[0], "K_A_Gbar (Mho/m^2)", 1, 100, valinit = 0) )
    #sliders[-1].on_changed( setK_A_Gbar )
    #ka_slider_index = len(sliders)-1

    sliders.append( Slider( axes[1], "Dynamic Inh. Delay", 1, 20.0, valinit = dynamicDelay, valfmt='%0.2f'))
    sliders[-1].on_changed( setDynamicDelay )
    dynDel_slider_index = len(sliders)-1

    #for j in sliders:
    #    j.label.set_fontsize(8)
    stim = Button( axStim, 'Spiking off', color = 'yellow' )
    stim.label.set_fontsize(10)
    stimObj = stimToggle( stim, axStim )
    
    #ka_current = Button( axKA, 'KA off', color = 'yellow' )
    #ka_current_obj= kaToggle( ka_current, axKA, ka_slider_index )
 
    load_button = Button( axLoad, 'Load File', color = 'yellow'  )
    load_button.label.set_fontsize(10)
    #load_obj= dynamicDelayToggle( load_button, axLoad, dynDel_slider_index)
    load_obj= fileDialog(load_button) #load_button, axLoad, dynDel_slider_index)
 
    reset = Button( axReset, 'Reset', color = 'cyan' )
    reset.label.set_fontsize(10)
    q = Button( axQuit, 'Quit', color = 'pink' )
    q.label.set_fontsize(10)


    #sliders.append( Slider( axes[3], "GABA Onset time (ms)", 10, 50, valinit = gabaOnset * 1000) )
    #sliders[-1].on_changed( setGabaOnset )
    #sliders.append( Slider( axes[4], "RM (Ohm.m^2)", 0.1, 10, valinit = RM))
    #sliders[-1].on_changed( setRM )
    #sliders.append( Slider( axes[5], "CM (Farads/m^2)", 0.001, 0.1, valinit = CM, valfmt='%0.3f'))
    #sliders[-1].on_changed( setCM )
    def resetParms( event ):
        for i in sliders:
            i.reset()
        reInitialize()

    def reInitialize():
        global sdn_x, sdn_y, gluGbar_list, gabaonset_list, max_g_exc
        sdn_x = []
        sdn_y = []
        IE_ratio_arr=[]
        max_g_exc, gluGbar_list, gabaonset_list, peaktime_list = [], [], [], []
        #dynDel_plot.set_data([[],[]])
        #dynDel_last_point.set_data([[],[]])
        sdn_plot.set_offsets(np.array([[],[]]))
        sdn_plot.set_array(np.array([]))
        del_exc_scat.set_xdata([])
        del_exc_scat.set_ydata([])
        #last_point.set_data([[],[]])
        #peaktime_plot.set_data([[],[]])
        #peaktime_last_point.set_data([[],[]])
        
    stim.on_clicked( stimObj.click )
    #ka_current.on_clicked(ka_current_obj.click)
    load_button.on_clicked( load_obj.click )
    reset.on_clicked( resetParms )
    q.on_clicked( doQuit )

    if printOutput:
        print( "maxval\tpk\tmaxt\tgluG\tgabaG\tK_A_G\tgabaon\tRM\tCM" )
    updateDisplay()

    plt.show()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
        main()
