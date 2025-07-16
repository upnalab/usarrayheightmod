import serial # pip install pyserial (not pip install serial)
import time
import ctypes
from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import assert_pico2000_ok, adc2mV
import matplotlib.pyplot as plt
import numpy as np

class utilsGCode:
    def __init__(self):
        # Initialize Serial
        self.s = None
        # Plot Parameters
        self.limits = 124 # Corner (maximum distance one can go in both X and Y)
        
        # Uncomment for plot the values and the wave section
        self.fig, self.ax = plt.subplots(1, 2, figsize=(16,9))
        
    def connectToMachine(self, relativeMode = True):
        self.s = serial.Serial("COM7", baudrate = 115200) # Connect to serial

        time.sleep(1) # Sleep one second so it doesn't get overbooked (maybe sometimes more time is needed)

        self.s.write( "G28\n".encode() ) # Go to the initial position

        print("Going to initial position")
        
        self.s.write( "M999\n".encode() ) # Reset temperature (needed for this machine)

        if relativeMode:
            self.s.write( "G91\n".encode() ) # Activate relative mode

        utilsGCode.waitForMoveToFinish(self, 'Ready To Start')
    
    def close(self):
        # Go to the inital position and close the port
        self.s.write( "G28\n".encode() )
        self.s.close()
        print('Execution has finish. Returning to the initial position')
        
    def move(self, X = 0, Y = 0, Z = 0, F = 10000, startMsg = None, endMsg = None):
        if startMsg != None:
            print(startMsg)
        
        self.s.write( ("G01 X" + str(X) + " Y" + str(Y) + " Z" + str(Z) + " F" + str(F) + "\n").encode() )
        
        utilsGCode.waitForMoveToFinish(self, endMsg)
        
    def waitForMoveToFinish(self, msgFinish):
        self.s.write( "M400 \n".encode() )
        self.s.write( ("M118 " +  str(msgFinish) + "\n").encode() )
        
        r = self.s.readline()
        while r != (str(msgFinish) + "\n").encode():
            r = self.s.readline()
        
        if msgFinish != None:
            print(msgFinish)
        
    def scatter(self, values, bufferA, voltsUnits = False):
        """
            Plot the values
        """
        # Erase previous plots
        # plt.clf()
        # # Plot the image
        # plt.imshow(values, origin='lower')
        # # # Add Colorbar
        # plt.colorbar()
        # # Pause plot so it can be shown
        # plt.pause(0.001)
        # # Show plot
        # plt.show()
        
        """
        Plot the values and the wave
        """
        # Erase previous plots
        self.ax[0].cla();self.ax[1].cla()
        
        
        times = range(utilsPico().maxSamples)
        self.ax[0].set_xlabel('time/ms')
        self.ax[0].set_ylabel('ADC')
        self.ax[0].set_title('Data read from the PicoScope')
        self.ax[0].plot(times, bufferA)
        self.ax[0].set_ylim([-40_000, 40_000])
        
        
        self.ax[1].imshow(values, origin='lower')
        # Colorbar
        # plt.colorbar()
        # Pause plot so it can be shown
        plt.pause(0.01)
        # Show plot
        plt.show()
        """
        Plot the wave
        """
        # Erase previous plots
        # plt.clf()
        
        
        # times = range(utilsPico().maxSamples)
        # plt.xlabel('time/ms')
        
        # plt.title('Data read from the PicoScope')
        # plt.plot(times, bufferA)
        # if not voltsUnits:
        #     plt.ylim([-25_000, 40_000])
        #     plt.ylabel('ADC')
        # else:
        #     plt.ylim([-10, 10])
        #     plt.ylabel('Volts')
        #     plt.suptitle(f'PeakToPeak = {values} V')
        # # Pause plot so it can be shown
        # plt.pause(0.01)
        # # Show plot
        # plt.show()
        

class utilsPico:
    def __init__(self):
        self.status = {}
        self.maxSamples = 3250
        self.oversample = ctypes.c_int16(1)  # Number of samples per aggregate, in our case 1
        self.timeIndisposedms = ctypes.c_int32()  # A pointer to the approximate time, in milliseconds, that the ADC (Analog to Digital Converter) will take to collect data
        self.maxValue = 32767
        self.rangePico = ps.PS2000_VOLTAGE_RANGE['PS2000_10V']
        
        # Maximun number of outliers
        self.maxOutliers = 10 # Needed if we use option 2 for computing peak2peak
    
    def openUnit(self):
        self.status["openUnit"] = ps.ps2000_open_unit()
        assert_pico2000_ok(self.status["openUnit"])  # We assert we can open the pico

        chandle = ctypes.c_int16(self.status["openUnit"])
        # Now let us set the channel A
        self.status["setChA"] = ps.ps2000_set_channel(chandle,
                                                 0,  # 0 means channel A, 1 means channel B
                                                 1,  # enable 1 == yes
                                                 1,  # Coupling type -> 0 = AC, 1 = DC
                                                 self.rangePico # Originally: ps.PS2000_VOLTAGE_RANGE['PS2000_100MV'],
                                                 
                                                 
                                                 # Range = 9 (that fix the scale in this case 9 means 10V
                                                 )
        # Pico ranges: Range = 1 -> PS2000_20mV; Range = 2 -> PS2000_50mV; Range = 3 -> PS2000_100mV; Range = 4 -> PS2000_200mV; Range = 5 -> PS2000_500mV;
        #              Range = 6 -> PS2000_1V; Range = 7 -> PS2000_2V; Range = 8 -> PS2000_5V; Range = 9 -> PS2000_10V; Range = 10 -> PS2000_20V;
        assert_pico2000_ok(self.status["setChA"])
        
    def closeUnit(self):
        chandle = ctypes.c_int16(self.status["openUnit"])
        
        # Let us make sure that the pico stops and close after the execution
        self.status["stop"] = ps.ps2000_stop(chandle)
        assert_pico2000_ok(self.status["stop"])

        self.status["close"] = ps.ps2000_close_unit(chandle)
        assert_pico2000_ok(self.status["close"])
        
    def blockMode(self, returnV):
        chandle = ctypes.c_int16(self.status["openUnit"])
        # Run a Block -> ps2000_run_block tells the oscilloscope to start collecting data in block mode
        self.status["runBlock"] = ps.ps2000_run_block(chandle,
                                                 self.maxSamples,
                                                 5,
                                                 self.oversample,
                                                 ctypes.byref(self.timeIndisposedms))
        assert_pico2000_ok(self.status["runBlock"])

        # Collect Data -> In order to h¡get the data we need to use ps2000_get_values
        # Create buffers ready for data
        bufferA = (ctypes.c_int16 * self.maxSamples)()
        cmaxSamples = ctypes.c_int32(self.maxSamples)
        
        while ps.ps2000_ready(chandle) <= 0:
            hasCollectData = False
            pass
        
        hasCollectData = True
        self.status["getValues"] = ps.ps2000_get_values(chandle,
                                                   ctypes.byref(bufferA),  # Pointer to write the data of Channel A
                                                   None,  # Nothing will be written in Channel B as we are not going to use it
                                                   None,  # Nothing will be written in Channel C as we are not going to use it
                                                   None,  # Nothing will be written in Channel D as we are not going to use it
                                                   ctypes.byref(self.oversample),
                                                   # pointer_to_overflow (If the pointer is 0 -> Channel A overflow, if it is 1 -> Channel B, etc.
                                                   cmaxSamples)
        assert_pico2000_ok(self.status["getValues"])
        
        if returnV:
            # find maximum ADC count value
            maxADC = ctypes.c_int16(self.maxValue)  # The oscilloscope will return data as ADC values,which range from -32767 to +32767. These need to be converted with adc2mV.
                                            # This function will accept the buffer_a, and output a list of equivalent values in mV
    
            
            # convert ADC counts data to mV
            adc2mVChA = adc2mV(bufferA,
                               self.rangePico,
                               maxADC)
            return np.array(adc2mVChA), hasCollectData
        
        return bufferA, hasCollectData
    
    def collectData(self, returnV = False):
        Peak2peak = []
        for i in range(10):
            while True:
                data, hasCollectData = self.blockMode(returnV)
                if hasCollectData:
                    """
                        3 different options to compute peak2peak 
                    """
                    # # Option 1: Simple Peak2peak
                    peak2peak = max(data) - min(data)
                        
                    # # Option 2: As there might be outliers instead of picking up the max and min value, we choose the self.maxOutliers highest and lowest value
                    # orderData = np.sort(data)
                    # peak2peak = orderData[-self.maxOutliers] - orderData[self.maxOutliers]
                    
                    # Option 3: Using the fft
                    # fft = np.abs(np.fft.fft(data))
                    #     # 3 max freq are going to appear in the fft, 1 around 0 (noise) and the other two will be the same around 40Hz and -40Hz (because of simmetry). This one is our sought-after amplitude
                    # top3freq = np.sort(fft)[::-1][:3]
                    # if top3freq[0] == top3freq[1]:
                    #     peak2peak = top3freq[0]
                    # elif top3freq[0] == top3freq[2]:
                    #     peak2peak = top3freq[0]
                    # else:
                    #     peak2peak = top3freq[1] # Must be equivalent: peak2peak = top3freq[2]
                    
                    if peak2peak == 2 * self.maxValue:
                        print('Maximum value has been reached')
                    
                    Peak2peak.append(peak2peak)
                    break
        peak2peak = max(Peak2peak)
        return peak2peak, data

