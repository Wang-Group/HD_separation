from win32com import client
import datetime
import time
import logging
import numpy as np
import pandas as pd
import os
from EC_tool.algorithm.data_analysis import process_lsv_data, process_eis_data



eight_channel_id = {"channel1":{"port":"192.168.192.7:1031","SN":"CS310X24020509"},
                    "channel2":{"port":"192.168.192.7:1032","SN":"CS310X24020510"},
                    "channel3":{"port":"192.168.192.7:1033","SN":"CS310X24020511"},
                    "channel4":{"port":"192.168.192.7:1034","SN":"CS310X24020512"},
                    "channel5":{"port":"192.168.192.7:1035","SN":"CS310X24020513"},
                    "channel6":{"port":"192.168.192.7:1036","SN":"CS310X24020514"},
                    "channel7":{"port":"192.168.192.7:1037","SN":"CS310X24020515"},
                    "channel8":{"port":"192.168.192.7:1038","SN":"CS310X24020516"}}


class ElectroStation:

    is_connected: bool = False

    def __init__(self,channel: int, logger = logging.Logger):

        self.logger = logger

        self.client = client.Dispatch("{82423ECF-EFC0-47E0-8AB0-35ADE2CD3EEF}")
        
        self.connect(channel_num=channel)

        # initialization parameters
        self.client.SetGNDMode(isVirMode=True)

        self.client.SetEarthMode(isSet=False)


    def connect(self, channel_num: int):
        # connect the channel
        self.client.FindDeviceByName(eight_channel_id[f'channel{channel_num}']['port'])
      
        if self.client.GetSerialNumber() == eight_channel_id[f'channel{channel_num}']['SN']:
            print(f'channel{channel_num}：Electrochemical workstation is connected')
            time.sleep(0.5)
            self.logger.info(f'channel{channel_num}：Electrochemical workstation is connected')
        else:
            self.logger.info(f'channel{channel_num}：Electrochemical workstation connection failed!')
            raise ValueError(f"channel{channel_num}: Electrochemical workstation connection failed!")
        
        self.is_connected = self.client.DeviceIsOpen()


    def disconnect(self):
        self.client.DeviceDisconnect()


    def wait_for_test_end(self):
        while self.client.ExpIsEnd() is False:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt as e:
                self.client.DeviceStop()
                print('Stopped by Keyboard')
                return False
            except Exception as e:
                raise Exception(e)
            
        time.sleep(1)        
        print("test completed!")
        return True


    def clear_temp_data(self):
        self.client.ClearData()
        assert self.client.GetValues()[1] == (), "Fail to clear cache!"
        print("complete cache clearing!")


    def get_iR_drop(self, file_path = None, exp_index = None, iR_comp_coef = 0.85):

        assert self.client.DeviceIsOpen()
        self.clear_temp_data()

        time.sleep(1)

        if self.client.SetCurrentRange(isAutoRange=True, nRange=0, nMinRange = 6, nMaxRange = 2, isOnlyInc = True):
            self.logger.info(f'channel：current range initialization')
            print("Current range initialization is complete!")
        
        if self.client.SetVoltRange(isAuto=True, nRange=2):
            self.logger.info(f'channel: voltage range initialization')
            print("Voltage range initialization is complete!")

        time.sleep(3)

        self.client.SetGNDMode(isVirMode=True)
        self.client.SetEarthMode(isSet=False)

        self.client.SetEISVSTimeParams(fFreq = 10000,  # Measurement frequency, Hz
                                       fSineBias = 0,  # DC voltage, V
                                       fSineAmp = 50,  # AC amplitude, mV
                                       fPeriod = 1,  # Measurement interval, V
                                       fTime = 5  # Total measurement time, s
                                       )
        
        print("Start the ohmic drop compensation test!")
        time.sleep(1)
        self.client.DeviceStart()
        self.wait_for_test_end()

        time.sleep(2)
        
        eis_results = np.array(self.client.GetEisValues()[4]).T
        res = eis_results.mean()
        print(res)

        time.sleep(2)

        # Sometimes, impedance testing fails and returns np.nan values, resulting in experimental failure!
        i=2
        while (np.isnan(res)) and (i <= 3):
            self.clear_temp_data()
            time.sleep(1)
            self.client.SetEISVSTimeParams(fFreq = 10000, fSineBias = 0, fSineAmp = 50, fPeriod = 1, fTime = 10 )
            time.sleep(1)
            self.client.DeviceStart()
            self.wait_for_test_end()
            time.sleep(2)
            eis_results = np.array(self.client.GetEisValues()[4]).T
            res = eis_results.mean()
            
            i = i+1

        if np.isnan(res) or res > 50:
            raise ValueError(f"channel{self.channel}: Incorrect internal resistance test! Please check the circuit!")
        
        iR_comp_value = np.round(res * iR_comp_coef, 2)
        print(f"系统内阻测试完成:\n\t测试电位: 0 V vs Ref\n\t测试内阻: {res} \n\tiR补偿系数: {iR_comp_coef*100} %\n\tiR补偿: {iR_comp_value}")
        
        return iR_comp_value, res
    
    
    def run_lsv_test(self,
                     final_potential,
                     initial_potential,
                     scan_rate,  # mV/s
                     iR_comp = False,
                     iR_comp_coef = 0.85, 
                     exp_index = None,
                     file_path = None):
        
        # Ohmic drop test
        if iR_comp == True:
            iR_comp_value, res = self.get_iR_drop(file_path = file_path, exp_index = exp_index, iR_comp_coef = iR_comp_coef)
            self.client.SetIRValue(isEnable=True, fValue=res)
        else:
            pass
            iR_comp_value, res = None, None
            self.client.SetIRValue(isEnable=False, fValue=0) 

        # LSV test
        assert self.client.DeviceIsOpen()
        self.clear_temp_data()
        time.sleep(3)

        if self.client.SetCurrentRange(isAutoRange=False, nRange = 0, nMinRange = 6, nMaxRange = 0, isOnlyInc = True):
            self.logger.info(f'channel：current range initialization')
            print("Current range initialization is complete!")
        
        if self.client.SetVoltRange(isAuto=False, nRange=2):
            self.logger.info(f'channel: voltage range initialization')
            print("Voltage range initialization is complete!")
            time.sleep(3)

        self.client.SetLSVAParams(fWaitTime = 5,
                                  fInitE = initial_potential,
                                  fFinalE = final_potential,
                                  fScanRate = scan_rate / 1000,
                                  fFrq = scan_rate / 1000 * 500)  # 2mV采样一次
        time.sleep(2)

        self.client.DeviceStart()
        self.wait_for_test_end()
        time.sleep(2)

        return iR_comp_value, res

    def run_cv_test(self,
                    lower_potential,
                    upper_vertex_potential,
                    initial_potential,
                    segments,  # 2 Segments every Cycle
                    scan_rate,  # mV/s
                    sweep_direction = False,  # False is negative，True is positive
                    iR_comp = False,
                    iR_comp_coef = 0.85,
                    save_path = "./tmp/CV"):
        # Ohmic drop test
        if iR_comp == True:
            iR_comp_value, res = self.get_iR_drop(iR_comp_coef)
            self.client.SetIRValue(isEnable=True, fValue=res)
        else:
            iR_comp_value, res = None, None
            self.client.SetIRValue(isEnable=False, fValue=0)

        # cv test
        assert self.client.DeviceIsOpen()
        self.clear_temp_data()
        time.sleep(2)

        if self.client.SetCurrentRange(isAutoRange=False, nRange=0, nMinRange = 6, nMaxRange = 1, isOnlyInc = True):
            self.logger.info(f'channel：current range initialization')
            print("Current range initialization is complete!")
        
        if self.client.SetVoltRange(isAuto=False, nRange=2):
            self.logger.info(f'channel: voltage range initialization')
            print("Voltage range initialization is complete!")
        time.sleep(3)

        if sweep_direction:
            print("zhengsao ")
            self.client.SetCVParams(fWaitTime = 5, 
                                    fInitE = initial_potential,
                                    fStepE1 = initial_potential,
                                    fStepE2 = upper_vertex_potential,
                                    fFinalE = initial_potential,
                                    fScanRate = scan_rate / 1000,
                                    fFrq = scan_rate / 1000 * 500,
                                    nCycles = int(segments / 2))
            
        else:
            print("fansao ")
            self.client.SetCVParams(fWaitTime = 5,
                                    fFrq = scan_rate / 1000 * 500,
                                    fInitE = initial_potential,
                                    fStepE1 = initial_potential,
                                    fStepE2 = lower_potential,
                                    fFinalE = initial_potential,
                                    fScanRate = scan_rate / 1000,
                                    nCycles = int(segments / 2))
            
            
            
        time.sleep(1)
        self.client.DeviceStart()
        self.wait_for_test_end()
        time.sleep(2)

        return iR_comp_value, res


    def run_it_test(self, 
                    potential,  # V
                    test_time,  # s
                    iR_comp = False,
                    iR_comp_coef = 0.85,
                    frq  = 5  # Hz, 1秒采集5个点
                    ):
        # Ohmic drop test
        if iR_comp == True:
            iR_comp_value, res = self.get_iR_drop(iR_comp_coef)
            self.client.SetIRValue(isEnable=True, fValue=res)
        else:
            iR_comp_value, res = None, None
            self.client.SetIRValue(isEnable=False, fValue=0)

        #  it test
        assert self.client.DeviceIsOpen()
        self.clear_temp_data()

        time.sleep(1)

        if self.client.SetCurrentRange(isAutoRange=False, nRange=0, nMinRange = 6, nMaxRange = 0, isOnlyInc = True):
            self.logger.info(f'channel：current range initialization')
            print("Current range initialization is complete!")
        
        if self.client.SetVoltRange(isAuto=False, nRange=0):
            self.logger.info(f'channel: voltage range initialization')
            print("Voltage range initialization is complete!")
        time.sleep(3)

        self.client.SetPotStaticParams(fWaitTime = 5,
                                       fPotValue = potential,
                                       fTestTime = test_time,
                                       fFreq = frq)
        time.sleep(1)

        self.client.DeviceStart()
        self.wait_for_test_end()
        time.sleep(1)
        
        return iR_comp_value, res

    def run_Et_test(self,
                    current: float,  # mA
                    test_time: float,  # s
                    iR_comp = False,
                    iR_comp_coef = 0.9,
                    frq  = 5,  # Hz, 1秒采集5个点
                    ):
        # Ohmic drop test
        if iR_comp == True:
            iR_comp_value, res = self.get_iR_drop(iR_comp_coef)
            self.client.SetIRValue(isEnable=True, fValue=res)
        else:
            iR_comp_value, res = None, None
            self.client.SetIRValue(isEnable=False, fValue=0)

        # Et test
        assert self.client.DeviceIsOpen()
        self.clear_temp_data()

        if self.client.SetCurrentRange(isAutoRange=False, nRange=0, nMinRange = 6, nMaxRange = 0, isOnlyInc = True):
            self.logger.info(f'channel：current range initialization')
            print("Current range initialization is complete!")
        
        if self.client.SetVoltRange(isAuto=False, nRange = 0):
            self.logger.info(f'channel: voltage range initialization')
            print("Voltage range initialization is complete!")
            time.sleep(3)

        self.client.SetGalStaticParams(fWaitTime = 5,
                                       fCurrentValue = current / 1000,
                                       fTestTime = test_time,
                                       fFreq = frq)
        time.sleep(1)
        self.client.DeviceStart()
        self.wait_for_test_end()
        time.sleep(1)
        
        return iR_comp_value, res
    
    def run_EIS_test(self, 
                     initial_freq: float = 100000,
                     final_freq: float = 0.01, 
                     potential: float = -1,
                     Ac_amplitude: float = 5, 
                     points_per_decade: int = 10, 
                     iR_comp = False,
                     iR_comp_coef = 0.85):
        # Ohmic drop test
        if iR_comp == True:
            iR_comp_value, res= self.get_iR_drop(iR_comp_coef)
            self.client.SetIRValue(isEnable=True, fValue=res)
        else:
            iR_comp_value, res = None, None
            self.client.SetIRValue(isEnable=False, fValue=0)

        # EIS test
        assert self.client.DeviceIsOpen()
        self.clear_temp_data()

        time.sleep(1)

        if self.client.SetCurrentRange(isAutoRange=True, nRange=0, nMinRange = 6, nMaxRange = 1, isOnlyInc = True):
            self.logger.info(f'channel：current range initialization')
            print("Current range initialization is complete!")
        
        if self.client.SetVoltRange(isAuto=True, nRange = 2):
            self.logger.info(f'channel: voltage range initialization')
            print("Voltage range initialization is complete!")
        time.sleep(3)

        self.client.SetGNDMode(isVirMode=True)
        self.client.SetEarthMode(isSet=False)
        time.sleep(1)

        self.client.SetEISVSfrqParams(fInitFreq = initial_freq,
                                      fFinalFreq = final_freq,
                                      fSineBias = potential,
                                      fSineAmp = Ac_amplitude,
                                      nPoints = int((np.log10(initial_freq) - np.log10(final_freq)) * points_per_decade))
        time.sleep(1)
        self.client.DeviceStart()
        self.wait_for_test_end()
        time.sleep(1)

        return iR_comp_value, res
    
    
    def save_path(self, save_path):
        """
        Create the folder for saving.
        """
        base_folder = os.path.abspath(save_path)
        folder_path = base_folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return os.path.abspath(folder_path)
    

    def save_result(self, exp_index, save_path):
        """
        save the result.
        """
        if self.client.GetValues()[0]:
            result = pd.DataFrame(np.array(self.client.GetValues()[1:]).T,
                                         columns=["Potential(V)", "Current(A)", "Time(s)"],)
            folder_path = self.save_path(save_path)
            file_name = f'{exp_index}.txt'
            save_path = os.path.join(folder_path, file_name)
            result.to_csv(save_path, sep='\t', index=False)
        time.sleep(2) 
        
        return save_path
    
    def save_eis_result(self, exp_index, save_path):
        """
        save the eis result.
        """
        if self.client.GetEisValues()[0]:
            result = pd.DataFrame(np.array(self.client.GetEisValues()[1:]).T,
                                         columns=["Frequency Value(Hz)", "Real Value(ohm)", "Imaginary Value(ohm)", "ZValue(ohm)", "Phase Angle Value(degree)"],)
            folder_path = self.save_path(save_path)
            file_name = f'{exp_index}.txt'
            save_path = os.path.join(folder_path, file_name)
            result.to_csv(save_path, sep='\t', index=False)
        time.sleep(2) 
        
        return save_path







    


