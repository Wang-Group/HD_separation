import pandas as pd
import numpy as np 
import logging
import json
import time
from datetime import datetime
import threading
import serial
from EC_tool.manager.pumps import PumpOperations
from EC_tool.manager.robotic_arm import RoboticPlatformMassHD
from EC_tool.manager.instrument import ElectroStation, MASS_wx
from EC_tool.manager.bopai_controller import Bopai
from EC_tool.manager.runze_driver import *
from EC_tool.manager.runze_ascii.syringe_pump import SyringePump_ASCII

from EC_tool.algorithm.data_analysis import process_lsv_data, process_eis_data, process_mass_standard_curve, process_mass_data



def load_json_file(path):
    """
    load in json file as configurtion files
    Args:
        path: the path to the json file
    Returns:
        the config
    """
    with open(path, "r", encoding='utf-8') as f:
        config = json.load(f)
    return config

class PlatformManager:
    def __init__(
            self, 
            config_folder: str,
            exp_path: str,
            mass_save_path,
            mass_calibration_save_path,
            logger: logging.Logger
            ):

        # load the config file
        self.config_folder = config_folder
        self.exp_path = exp_path
        self.mass_save_path = mass_save_path
        self.mass_calibration_save_path = mass_calibration_save_path
        self.logger = logger
        self.active_learning = False

        # initialize the locks
        self.arm_lock = threading.Lock()
        self.screen_lock = threading.Lock()
        self.excel_lock = threading.Lock()
        self.syn_flag_lock = threading.Lock()
        self.cat_flag_lock = threading.Lock()
        self.mass_flag_lock = threading.Lock()
        self.terminate_count_lock = threading.Lock()

        # initialize the flags and data
        self.synthesis_in_progress = False # flag to indicate if the synthesis is going on 
        self.catalysis_in_progress = False # flag to indicate if the catalysis is going on
        self.catalysis2_in_progress = False # flag to indicate if the catalysis2 is going on
        self.mass_in_progress = False # flag to indicate if the mass is going on
        self.exp_data = None   
        self.mass_file_name = None
        self.mass_time = None
        self.t_start = "2025/04/29 11:32:22"
        self.t_end = "2025/04/29 12:36:51"

    def initialize_devices(self, refill_cat_cell_flag = False):
        """
        Initialize the pumps, robotic arms, MS, and potentiostats.
        
        Args:
            refill_cat_cell_flag (bool): Flag to indicate if the catalysis cell should be refilled.
        """
        # initialize the pumps
        print("start pumps")
        pump_config = load_json_file(self.config_folder + "/pumps/config.json")
        self.pumps = PumpOperations(port_num1 = pump_config["port_num1"],
                                    port_num2 = pump_config["port_num2"],
                                    port_num3 = pump_config["port_num3"],
                                    channels = pump_config["channels"],
                                    commands = pump_config["commands"],
                                    switch_slave = int(pump_config["switch_slave"],16),
                                    syringe_slave = int(pump_config["syringe_slave"],16),
                                    syringe_slave2 = int(pump_config["syringe_slave2"],16),
                                    logger = self.logger)
        
        serial_0 = serial.Serial("COM22", 9600, timeout = 0.1)
        self.cell_pump = SyringePump_ASCII(serial_0, slave = 0, model = "SY-09S", volume = 8, logger = self.logger)

        serial1 = serial.Serial("COM20", 9600, timeout = 0.1)
        self.H2_pump = SyringePump(serial1, slave = 0, model = "SY-09-3-mL", volume = 3, logger = self.logger)
        self.H2_pump.forced_reset()

        serial2 = serial.Serial("COM21", 9600, timeout = 0.1)
        self.HD_pump = SyringePump(serial2, slave = 5, model = "SY-09-3-mL", volume = 3, logger = self.logger)
        self.HD_pump.forced_reset()

        serial3 = serial.Serial("COM19", 9600, timeout = 0.1)
        self.N2_pump = SyringePump(serial3, slave = 0, model = "SY-09-8-mL", volume = 8, logger = self.logger)
        self.N2_pump.forced_reset()
        
        # initialize the bopai_controller
        print("start bopai_controller")
        controller_config = load_json_file(self.config_folder + "/bopai/bopai.json")
        self.bopai_controller = Bopai(motion_prm = controller_config["bopai_controller"], 
                         logger = self.logger)
        self.bopai_controller.connection()

        # initialize the robotic arm
        print("start robotic arm")
        robotic_arm_config = load_json_file(self.config_folder + "/robotic_arm/config.json")
        self.robotic_platform = RoboticPlatformMassHD(rail_param = robotic_arm_config["rail_param"],
                                                arm_ip_address = robotic_arm_config["arm_ip_address"],
                                                arm_port = robotic_arm_config["arm_port"],
                                                logger = self.logger)
        
        print("start the instruments")
        # initialize the instruments
        # initialize the potentiostats
        self.channel1 =  ElectroStation(channel = 1, logger = self.logger)
        self.channel2 =  ElectroStation(channel = 2, logger = self.logger)
        self.channel3 =  ElectroStation(channel = 3, logger = self.logger)
        self.channel4 =  ElectroStation(channel = 4, logger = self.logger)
        self.channel5 =  ElectroStation(channel = 5, logger = self.logger)
        self.channel6 =  ElectroStation(channel = 6, logger = self.logger)
        self.channel7 =  ElectroStation(channel = 7, logger = self.logger)
        self.channel8 =  ElectroStation(channel = 8, logger = self.logger)

        # initialize the mass
        instrument_config = load_json_file(self.config_folder + "/instruments/config.json")
        self.mass = MASS_wx(app_path=instrument_config["MASS"]["app_path"], logger = self.logger)

        self.refill_cat_cell_flag = refill_cat_cell_flag

    
    def MS_calibration(self, t:int = 120):
        """
        Perform the MS calibration process.
        """

        for HD_volume in [3/3, 3/3,4/3, 5/3, 6/3, 7/3, 8/3, 9/3]:
            H2_volume = 12/3 - HD_volume

            self.HD_pump.set_dynamic_speed(speed = HD_volume/t)
            self.H2_pump.set_dynamic_speed(speed = H2_volume/t)

            self.HD_pump.open_valve()
            time.sleep(0.2)
            self.H2_pump.open_valve()
            time.sleep(0.2)

            if HD_volume == 3/3:
                self.t_start = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print(self.t_start)

            t1 = threading.Thread(target = self.HD_pump.move_to_absolute_position, args = (HD_volume, ))
            t2 = threading.Thread(target = self.H2_pump.move_to_absolute_position, args = (H2_volume, ))

            time1 = time.time()
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            time2 = time.time()

            print(time1-time2)

            self.HD_pump.close_valve()
            time.sleep(0.2)
            self.H2_pump.close_valve()
            time.sleep(0.2)
            
            t1 = threading.Thread(target = self.HD_pump.move_to_absolute_position, args = (0, ))
            t2 = threading.Thread(target = self.H2_pump.move_to_absolute_position, args = (0, ))

            time1 = time.time()
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            time2 = time.time()

            print(time1-time2)

            time.sleep(60*5)
        self.mass.save_file(lock = self.screen_lock)
        self.t_end = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print(self.t_end)

        # Turn off the gas source
        self.H2_pump.close_valve()
        self.HD_pump.close_valve()
        self.N2_pump.close_valve()

    def convert_to_bool(self, data):
        """
        Convert a data array to True, False, or None.
        
        Args:
            data (numpy array): A numpy array.
        
        Returns:
            list: Converted list with boolean or None values.
        """

        data_temp = []
        for i in data:
            if i == 1 or i == 0:
                data_temp.append(bool(i))
            elif i == None:
                data_temp.append("None")
            else:
                raise ValueError(f"the element should only be 1/0/None. current value is {i}")
        return data_temp

    def round_to_two_decimals(self, data):
        """
        Convert a data array to float with two decimals.
        
        Args:
            data (numpy array): A numpy array.
        
        Returns:
            list: Converted list with float values rounded to two decimals.
        """
        data_temp = []
        for i in data:
            if i is not None:
                data_temp.append(np.around(i,2))
            elif i is None:
                data_temp.append("None")
            else:
                raise ValueError(f"the element should only be None or float. current value is {i}")
        return data_temp

    def check_exp_file(self):
        """
        Load a local excel file and return its local copy.
        
        Returns:
            DataFrame: A copy of the experimental data.
        """

        # we should use the loack for self.exp_data and the excel file
        self.excel_lock.acquire()
        if self.exp_data is None:
            self.exp_data = pd.read_json(self.exp_path).replace({np.nan: None, "None": None})
            self.exp_data["volume1"] = self.round_to_two_decimals(self.exp_data["volume1"].values)
            self.exp_data["volume2"] = self.round_to_two_decimals(self.exp_data["volume2"].values)
            self.exp_data["volume3"] = self.round_to_two_decimals(self.exp_data["volume3"].values)
            
            self.exp_data["ir_value"] = self.round_to_two_decimals(self.exp_data["ir_value"].values)

            self.exp_data["syn_completed"] = self.convert_to_bool(self.exp_data["syn_completed"].values)
            self.exp_data["in_storage"] = self.convert_to_bool(self.exp_data["in_storage"].values)
            self.exp_data["cat_completed"] = self.convert_to_bool(self.exp_data["cat_completed"].values)
            
            self.exp_data = self.exp_data.replace({np.nan: None, "None": None})

        exp_data = self.exp_data.copy()
        self.excel_lock.release()
        return exp_data

    def terminate(self, terminate_num):
        """
        Determine whether to terminate the experiment based on the number of completed synthesis and catalysis experiments.

        Args:
            terminate_num (list): A list containing two integers. The first integer represents the number of synthesis experiments to be completed, and the second integer represents the number of catalysis experiments to be completed.

        Returns:
            bool: True if the experiment should be terminated, False otherwise.
        """
        # count how many catalysts are done and if we should stop now
        self.terminate_count_lock.acquire()
        if ((self.terminate_count[0] >= terminate_num[0]) and (self.terminate_count[1] >= terminate_num[1])):
            # we should stop now
            self.terminate_count_lock.release()
            # wait until both experiments are finished
            while True:
                self.cat_flag_lock.acquire()
                self.syn_flag_lock.acquire()
                # break the loop when both experiments are finished
                if (self.synthesis_in_progress is False) and (self.catalysis_in_progress is False) and (self.catalysis2_in_progress is False):
                    self.cat_flag_lock.release()
                    self.syn_flag_lock.release()                
                    break
                else:
                    self.cat_flag_lock.release()
                    self.syn_flag_lock.release()
                    time.sleep(10)
            return True
        else:
            self.terminate_count_lock.release()
            return False

    def run_exp_in_parallel(self, terminate_num = [12, 12]):
        """
        Run experiments in parallel until the specified number of synthesis and catalysis experiments are completed.
        
        Args:
            terminate_num (list): Number of synthesis and catalysis experiments in this run.
            reproduce_former_exp (bool): Flag to indicate if former experiments should be reproduced.
        """

        if self.refill_cat_cell_flag:

            # self.refill_cat_cell()
            # self.MS_calibration()
            self.Pt_test()

        # run the code to read in the excel content
        # if self.exp_data is not defined, it will be read from a local file.
        _ = self.check_exp_file()

        # run catlaysis_num amount of catalysis experiments
        self.terminate_count_lock.acquire()
        self.terminate_count = [0, 0]
        self.terminate_count_lock.release()

        # run the experiments all the way
        while True:
            # check if we should stop the experiments
            terminate = self.terminate(terminate_num = terminate_num)
            if terminate:
                break

            # check if synthesis is ongoing, if it is, we do nothing. if it is not, we will start a new experment or perform AL.
            self.syn_flag_lock.acquire()
            if (self.synthesis_in_progress is False):
                # record we are going to do synthesis now
                self.syn_flag_lock.release()
                # check if we got enough experiments to perform (e.g., there are more than 3 synthesis to do, and more than 3 free slots),
                # and got enough slots to store the catalysis
                self.excel_lock.acquire()
                if (sum(self.exp_data["syn_completed"].values == None)>3) and (sum(self.exp_data["in_storage"].values == True)<7):
                    self.excel_lock.release()
                    # check if we should stop because of the synthesis number
                    self.terminate_count_lock.acquire()
                    if self.terminate_count[0] >= terminate_num[0]:
                        # do nothing
                        self.terminate_count_lock.release()
                    else:
                        # execute synthesis
                        self.terminate_count_lock.release()
                        self.logger.info("going to perform synthesis")
                        f_syn = threading.Thread(target=self.run_synthesis)
                        f_syn.start()
                else:
                    self.excel_lock.release()
            else:
                # log that the synthesis is going on now
                self.syn_flag_lock.release()

            # get the locker to check if catalysis is going on now
            self.cat_flag_lock.acquire()
            if (self.catalysis_in_progress is False) or (self.catalysis2_in_progress is False):
                self.cat_flag_lock.release()
                # get the locker to actually execute the experiment, if there is no catalysis going on
                self.excel_lock.acquire()
                if sum(self.exp_data["in_storage"].values == True) > 0:
                    self.excel_lock.release()
                    # check if we should stop because of the catalysis number
                    self.terminate_count_lock.acquire()
                    if self.terminate_count[1] < terminate_num[1]: 
                        self.terminate_count_lock.release()
                        self.mass_flag_lock.acquire()
                        if (self.mass_in_progress is False):
                            # execute catalysis
                            self.mass_flag_lock.release()
                            self.logger.info("perform catalysis now")
                            f_cat = threading.Thread(target = self.run_catalysis)           
                            f_cat.start()
                        else:
                            self.mass_flag_lock.release()
                    else:
                        # do nothing
                        self.terminate_count_lock.release()
                else:  
                    # record we are not going to perform catalysis
                    self.excel_lock.release()
            else:
                self.cat_flag_lock.release()

            # wait for 10 seconds before next check
            time.sleep(10)
            
        # self.MS_calibration()  

    def extract_catalysis_infor(self):
        """
        Check the catalysis that should be executed.
        
        Returns:
            tuple: Experimental data, catalysis index, and storage location.
        """
        self.excel_lock.acquire()
        exp_data = self.exp_data.copy()
        self.excel_lock.release()

        cat_index = np.min(np.where(exp_data["in_storage"].values == True)[0])
        storage_location = int(exp_data["storage_location"].values[cat_index])

        return exp_data, cat_index, storage_location
    
    def MS_test(self):
        """
        Perform mass test by pumping in gas from the H cell.
        """
        for repeat in range(3):
            time.sleep(5*60)
            # if repeat == 0:
            #     self.mass_time_start = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

            self.cell_pump.set_dynamic_speed(speed = 14/3/120)
            self.cell_pump.switch_valve(port = 1)
            time.sleep(0.2)

            self.cell_pump.move_to_absolute_position(14/3, feedback = True)

            self.cell_pump.switch_valve(port = 0)
            time.sleep(0.2)

            self.cell_pump.move_to_absolute_position(0, feedback = True)

    def Et_test(self, channel, current = -50, test_time = 1600):
        """
        Perform Et test on channel 8.
        
        Args:
            current (int): Current value for the test.
            test_time (int): Duration of the test in seconds.
        """

        res, eis_results = channel.run_Et_test(current= current, test_time = test_time, iR_comp = True, frq = 5)
    
    def HD_test(self, cat_index, channel):
        """
        Perform HD test including Et test and MS test.
        """
        if channel == self.channel7:
            self.bopai_controller.cat_cell1(output="cat_cell")
        elif channel == self.channel8:
            self.bopai_controller.cat_cell2(output="cat_cell")

        self.mass_time_start = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        # save the data after completing the LSV test
        self.excel_lock.acquire()
        self.exp_data["mass_time_start"].values[cat_index] = self.mass_time_start
        self.exp_data.to_json(self.exp_path)
        # copy a data to parse to active learning
        data_to_pass = self.exp_data.copy()
        self.excel_lock.release()

        f1 = threading.Thread(target = self.Et_test, kwargs = {"channel": channel, "current":-50,  "test_time":1600} )
        f2 = threading.Thread(target = self.MS_test, )

        f1.start()
        f2.start()

        f1.join()
        f2.join()

        # data process
        channel.save_result(exp_index = cat_index, 
                                  save_path = f"./tmp/electrocatalysis/Et/")
        self.mass_flag_lock.acquire()
        self.mass_in_progress = False
        self.mass_flag_lock.release()

        self.bopai_controller.cat_cell1(output="cat_cell")

        # got IR drop with slow speed of 2 mV/s
        t1 = time.time()
        print(f"{cat_index}开始进行LSV2测试")
        iR_comp_value, res = channel.run_lsv_test(final_potential = -1.5,  # vs.Hg/HgO
                                                        initial_potential = -0.92,  # vs.Hg/HgO
                                                        scan_rate = 2,  # mV/s
                                                        iR_comp = False,
                                                        iR_comp_coef = 0.9,)
        
        file_path  = channel.save_result(exp_index=cat_index,
                                         save_path="./tmp/electrocatalysis/LSV/2")
        
        print(f"{cat_index}完成LSV2测试")
        
        # ensure there is minimal 2 minutes gap after the MS test 
        while time.time() - t1 <2*60:
            time.sleep(0.1)
        
        # process the mass data and get the separation_coefficient
        self.mass_time_end =  datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        # save the data after completing the LSV test
        self.excel_lock.acquire()
        self.exp_data["ir_value"].values[cat_index] = res
        # self.exp_data["mass_time_start"].values[cat_index] = self.mass_time_start
        self.exp_data["mass_time_end"].values[cat_index] = self.mass_time_end
        self.exp_data.to_json(self.exp_path)
        # copy a data to parse to active learning
        data_to_pass = self.exp_data.copy()
        self.excel_lock.release()
        
        # process the lsv data and get the potential
        over_potential_current = [-0.010, -0.020, -0.030, -0.040, -0.050]
        potential = [process_lsv_data(file_path, cat_index, mediana=i) for i in over_potential_current]
        # process the mass data and get the separation_coefficient
        H2_intercept, H2_slope, HD_intercept, HD_slope = process_mass_standard_curve(file_path = self.mass_calibration_save_path,
                                                                                     t_start = self.t_start, 
                                                                                     t_end = self.t_end)
        
        separation_coefficient = process_mass_data(file_path = self.mass_save_path, 
                                                   t_start = self.mass_time_start,
                                                   t_end = self.mass_time_end,
                                                   H2_intercept = H2_intercept, 
                                                   H2_slope = H2_slope,
                                                   HD_intercept = HD_intercept,
                                                   HD_slope = HD_slope)
        self.logger.info(f"{cat_index} separation coefficient is {separation_coefficient}, overpotential is {potential}")

        # save the data after completing the LSV test
        self.excel_lock.acquire()
        # get the overpotential for each different current
        for i in range(len(over_potential_current)):
            current = over_potential_current[i]
            self.exp_data[f"lsv_overpotential_{current}"].values[cat_index] = potential[i] 
        # self.exp_data["separation_coefficient"].values[cat_index] = separation_coefficient
        self.exp_data.to_json(self.exp_path)
        # copy a data to parse to active learning
        data_to_pass = self.exp_data.copy()
        self.excel_lock.release()
           
        # EIS test
        print(f"{cat_index} start eis testing")
        res, eis_results = channel.run_EIS_test(initial_freq = 100000,
                                                      final_freq = 0.1,
                                                      potential = potential[0],
                                                      Ac_amplitude = 5,
                                                      points_per_decade = 10,
                                                      iR_comp = False,
                                                      iR_comp_coef = 0.85,
                                                      )
        channel.save_eis_result(exp_index = cat_index, save_path = "./tmp/electrocatalysis/EIS")
        print(f"{cat_index} completed eis test")

        # CV test [-0.92 - 0.18]
        print(f"{cat_index} start cv200 testing")
        # fast scanning at 200 mV/s
        res, eis_results = channel.run_cv_test(lower_potential = -0.92,
                                                     upper_vertex_potential = 0.18,
                                                     initial_potential = -0.92,
                                                     segments = 10,  # 2 Segments every Cycle
                                                     scan_rate = 200,  # mV/s
                                                     sweep_direction = True,  # False is negative，True is positive
                                                     iR_comp = False,
                                                     iR_comp_coef = 0.85,
                                                     )
        channel.save_result(exp_index=cat_index, save_path="./tmp/electrocatalysis/CV/200")

        print(f"{cat_index} complete cv200 test")

        print(f"{cat_index} start cv100 testing")
        # 100 mV/s
        res, eis_results = channel.run_cv_test(lower_potential = -0.92,
                                                     upper_vertex_potential = 0.18,
                                                     initial_potential = -0.92,
                                                     segments=4,  # 2 Segments every Cycle
                                                     scan_rate=100,  # mV/s
                                                     sweep_direction=True,  # False is negative，True is positive
                                                     iR_comp=False,
                                                     iR_comp_coef=0.9,
                                                     )
        channel.save_result(exp_index=cat_index, save_path="./tmp/electrocatalysis/CV/100")
        print(f"{cat_index} complete cv100 test")


        # 50 mV/s
        res, eis_results = channel.run_cv_test(lower_potential = -0.92,
                                                     upper_vertex_potential = 0.18,
                                                     initial_potential = -0.92,
                                                     segments=4,  # 2 Segments every Cycle
                                                     scan_rate=50,  # mV/s
                                                     sweep_direction=True,  # False is negative，True is positive
                                                     iR_comp=False,
                                                     iR_comp_coef=0.9,
                                                     )
        channel.save_result(exp_index=cat_index, save_path="./tmp/electrocatalysis/CV/50")
        print(f"{cat_index} complete cv50 test")
        
        # 20 mV/s
        res, eis_results = channel.run_cv_test(lower_potential = -0.92,
                                                     upper_vertex_potential = 0.18,
                                                     initial_potential = -0.92,
                                                     segments=4,  # 2 Segments every Cycle
                                                     scan_rate=20,  # mV/s
                                                     sweep_direction=True,  # False is negative，True is positive
                                                     iR_comp=False,
                                                     iR_comp_coef=0.9,
        ) 
        channel.save_result(exp_index=cat_index, save_path="./tmp/electrocatalysis/CV/20")
        print(f"{channel} complete cv20 test")        
        return
            
    def run_catalysis(self):
        """
        Run catalysis experiment.
        """

        self.mass_flag_lock.acquire()
        self.mass_in_progress = True
        self.mass_flag_lock.release()

        # get the index of the experiment for catalysis
        exp_data, cat_index, storage_location = self.extract_catalysis_infor()

        self.cat_flag_lock.acquire()
        if (self.catalysis_in_progress is False):
            # change the flag so that it is True, which means we starterd a synthesis experiment
            self.catalysis_in_progress = True
            channel = self.channel7
            cell = 1

        elif (self.catalysis2_in_progress is False):
            # change the flag so that it is True, which means we starterd a synthesis experiment
            self.catalysis2_in_progress = True
            channel = self.channel8
            cell = 2

        self.cat_flag_lock.release()

            
        if cell == 1:
        # perform electrochemical test  
            self.robotic_platform.led_to_HD_test_cell1(lock = self.arm_lock,
                                                    dry_coil_address = 70 + storage_location)
        else:
            self.robotic_platform.led_to_HD_test_cell2(lock = self.arm_lock,
                                                    dry_coil_address = 70 + storage_location)
        
        # save the data after removing the catalyst from the storage
        self.excel_lock.acquire()
        self.exp_data["in_storage"].values[cat_index] = False
        self.exp_data.to_json(self.exp_path)
        self.excel_lock.release()
        
        #  electrochemical test 
        self.HD_test(channel= channel, cat_index = cat_index)

        if cell == 1:
            # discard the catalyst to the bin
            self.robotic_platform.HD_cell1_to_waste(lock = self.arm_lock)
        else:
            self.robotic_platform.HD_cell2_to_waste(lock = self.arm_lock)

        # save data for the completion of the catalysis
        self.excel_lock.acquire()
        self.exp_data["cat_completed"].values[cat_index] = True
        self.exp_data.to_json(self.exp_path)
        self.excel_lock.release()
        
        # record the success of one catalyst  
        self.terminate_count_lock.acquire()
        self.terminate_count[1] = self.terminate_count[1] + 1
        self.terminate_count_lock.release()

        # release the catalysis flag 
        self.cat_flag_lock.acquire()
        if cell == 1:
            self.catalysis_in_progress = False
        else:
            self.catalysis2_in_progress = False
        self.cat_flag_lock.release()

    def extract_synthesis_infor(self, batch_num):
        """
        Extract synthesis information.
        
        Args:
            batch_num (int): Number of experiments in a batch.
        
        Returns:
            tuple: Experimental data and synthesis information.
        """
        # get the information for the synthesis
        self.excel_lock.acquire()
        exp_data = self.exp_data.copy()
        self.excel_lock.release()

        syn_index = np.min(np.where(exp_data["syn_completed"].values != True)[0])

        substance1 = exp_data["substance1"].values[syn_index:syn_index+batch_num]
        substance2 = exp_data["substance2"].values[syn_index:syn_index+batch_num]
        substance3 = exp_data["substance3"].values[syn_index:syn_index+batch_num]
   
        volume1 = exp_data["volume1"].values[syn_index:syn_index+batch_num]
        volume2 = exp_data["volume2"].values[syn_index:syn_index+batch_num]
        volume3 = exp_data["volume3"].values[syn_index:syn_index+batch_num]
      

        return exp_data, syn_index, substance1, substance2, substance3, volume1, volume2, volume3

    def prepare_synthesis(self, batch_num):
        """
        Prepare synthesis recipe.
        
        Args:
            batch_num (int): Number of experiments in a batch.
        
        Returns:
            tuple: Experimental data and synthesis information.
        """
        # wait for 5 seconds before we start synthesis, so that the catalysis process has priority to use the arm
        time.sleep(5)

        # get the experiments to be executed
        exp_data, syn_index, substance1, substance2, substance3, volume1, volume2, volume3 = self.extract_synthesis_infor(batch_num)

        # check if the experiments are executed in batch mode
        if int(syn_index / batch_num) != syn_index / batch_num:
            raise ValueError(
                f"experiments should be executed as batch! \n syn_index: {syn_index} batch_num: {batch_num}")

        return exp_data, syn_index, substance1, substance2, substance3, volume1, volume2, volume3

    def electrodeposition(self, channel_port, exp_index):
        """
        Perform electrodeposition on a given channel.
        
        Args:
            channel_port (ElectroStation): Channel for electrodeposition.
            exp_index (int): Index of the experiment.
        """
        channel_port.run_Et_test(current = -300,  # mA
                                 test_time = 1200,  # s
                                 iR_comp = False,
                                 iR_comp_coef = 0.85,
                                 frq  = 5  # Hz, 1秒采集5个点
                                 )
        channel_port.save_result(exp_index = exp_index,
                                 save_path = "./tmp/electrodeposition")

    def run_synthesis(self, batch_num = 4):
        """
        Run synthesis experiment.
        
        Args:
            reproduce_former_exp (bool): Flag to indicate if former experiments should be reproduced.
            batch_num (int): Number of experiments in a batch.
        """
        # change the flag so that it is True, which means we starterd a synthesis experiment
        self.syn_flag_lock.acquire()
        self.synthesis_in_progress = True
        self.syn_flag_lock.release()

        # prepare the synthesis recipe
        exp_data, syn_index, substance1, substance2, substance3, volume1, volume2, volume3 = self.prepare_synthesis(batch_num)
        self.logger.info("finish preparation for synthesis")

        self.bopai_controller.empty_the_cell(output = "electrodeposition_cell_peristaltic_pump", time_sleep = 20)
        # perform electrodeposition
        self.robotic_platform.storage_to_electrodeposition(lock = self.arm_lock,
                                                           batch_num = batch_num)

        # mix solutions
        for i in range(batch_num):

            # add Na3Cit to the cell
            self.pumps.pipette_liquids(
                volume = volume3[i],
                channel = substance3[i],
                cell_channel = f"cell{i}")


            # add metal solution to the cell
            if volume1[i] < volume2[i]:
                self.pumps.pipette_liquids(volume = volume1[i],
                                           channel = substance1[i],
                                           cell_channel = f"cell{i}")
                self.pumps.pipette_liquids(volume = volume2[i],
                                           channel = substance2[i],
                                           cell_channel = f"cell{i}")
            else:
                self.pumps.pipette_liquids(volume = volume2[i],
                                           channel = substance2[i],
                                           cell_channel = f"cell{i}")
                self.pumps.pipette_liquids(volume = volume1[i],
                                           channel = substance1[i],
                                           cell_channel = f"cell{i}")
                

        # preparation catalys
        threads = []
        for i, channel in enumerate([self.channel3, self.channel4, self.channel5, self.channel6]):
            thread = threading.Thread(target = self.electrodeposition, args=(channel,syn_index + i,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        # clean the cells
        self.clean_syn_cell(syn_index)

        # record the success of four synthesis
        self.terminate_count_lock.acquire()
        self.terminate_count[0] = self.terminate_count[0] + 4
        self.terminate_count_lock.release()

        # change the flag so that it is false, meaning the completion of the synthesis
        self.syn_flag_lock.acquire()
        self.synthesis_in_progress = False
        self.syn_flag_lock.release()

    def clean_the_catalyst(self, 
                           repeat_num: int = 4,
                           reagent_name: str = "water", 
                           reagent_volume: float = 4.0,
                           in_speed: int = 200,
                           out_speed: int = 500):
        """
        Clean the catalyst.
        
        Args:
            repeat_num (int): Number of repetitions.
            reagent_name (str): Name of the cleaning reagent.
            reagent_volume (float): Volume of the cleaning reagent.
            in_speed (int): Speed of the inlet pump.
            out_speed (int): Speed of the outlet pump.
        """
        time.sleep(15)
        for repeat in range(repeat_num):
            self.pumps.dispense_liquid(volume = reagent_volume,
                                       inlet_channel = reagent_name,
                                       outlet_channel = "clean2",
                                       in_speed = in_speed,
                                       out_speed = out_speed)
            self.pumps.dispense_liquid(volume = reagent_volume,
                                       inlet_channel = reagent_name,
                                       outlet_channel = "clean1",
                                       in_speed = in_speed,
                                       out_speed = out_speed)
        for repeat in range(repeat_num):
            self.pumps.dispense_liquid(volume = reagent_volume,
                                        inlet_channel = reagent_name,
                                        outlet_channel = "clean2",
                                        in_speed = in_speed,
                                        out_speed = 650)

    def clean_syn_cell(self, 
                       syn_index: int, 
                       repeat_num: int = 5, 
                       batch_num: int = 4, 
                       reagent_name: str = "water", 
                       reagent_volume: float = 22.0,
                       in_speed: int = 800,
                       out_speed: int = 800):
        """
        Clean the synthesis cells with reagent.
        
        Args:
            syn_index (int): Index of the synthesis experiment.
            repeat_num (int): Number of repetitions.
            batch_num (int): Number of experiments in a batch.
            reagent_name (str): Name of the cleaning reagent.
            reagent_volume (float): Volume of the cleaning reagent.
            in_speed (int): Speed of the inlet pump.
            out_speed (int): Speed of the outlet pump.
        """
        self.check_exp_file()
        # rinse the tube so it is full of the cleaning reagent
        self.bopai_controller.empty_the_cell(output = "electrodeposition_cell_peristaltic_pump", time_sleep =20)

        # fill the com part
        self.pumps.dispense_liquid(volume = 3,
                                   inlet_channel = reagent_name,
                                   outlet_channel = "waste",
                                   in_speed = in_speed,
                                   out_speed = out_speed)

        # add water to the cells and wash the cells firstly
        for i in range(batch_num):
            self.pumps.dispense_liquid(volume = reagent_volume - 2,
                                       inlet_channel = reagent_name,
                                       outlet_channel = f"cell{i}",
                                       in_speed = in_speed,
                                       out_speed = out_speed)
        self.bopai_controller.empty_the_cell(output = "electrodeposition_cell_peristaltic_pump", time_sleep = 20)

        # wash the cells second time
        for i in range(batch_num):
            self.pumps.dispense_liquid(volume = reagent_volume - 2,
                                       inlet_channel = reagent_name,
                                       outlet_channel = f"cell{i}",
                                       in_speed = in_speed,
                                       out_speed = out_speed)

        self.robotic_platform.open_the_leds_to_dry()
        # move the catalysts to storage
        self.arm_lock.acquire()
        for i in range(batch_num):
            self.logger.info("moving the catalysts to the storage")
            storge_location = (syn_index + i)%8 # the total number of the storage location is assumed to be 8

            self.robotic_platform.electrodeposition_to_clean(catalyst_coil_address = 6 + i)
            f = threading.Thread(target = self.clean_the_catalyst)
            f.start()
            self.robotic_platform.robot_operation(96)
            f.join()
            self.robotic_platform.clean_to_dry(dry_coil_address = 11 + storge_location)

            # update the information for exp_data after completing the reaction and moving the catalysts to storage
            self.excel_lock.acquire()
            self.exp_data["syn_completed"].values[syn_index + i] = True
            self.exp_data["storage_location"].values[syn_index + i] = storge_location
            self.exp_data["in_storage"].values[syn_index + i] = True
            self.exp_data.to_json(self.exp_path)
            self.excel_lock.release()
        self.robotic_platform.slide_rail.home()
        self.arm_lock.release()

        self.robotic_platform.close_the_leds()

        self.bopai_controller.empty_the_cell(output = "electrodeposition_cell_peristaltic_pump", time_sleep = 20)
        
        # use H2SO4 to fill the com part
        self.pumps.dispense_liquid(volume = 3,
                                   inlet_channel = "H2SO4",
                                   outlet_channel = "waste",
                                   in_speed = in_speed,
                                   out_speed = out_speed)

        # add acid to the cells
        for i in range(batch_num):
            self.pumps.dispense_liquid(volume = reagent_volume - 2,
                                       inlet_channel = "H2SO4",
                                       outlet_channel = f"cell{i}",
                                       in_speed = in_speed,
                                       out_speed = 200)
        for i in range(batch_num):
            for repeat in range(5):
                self.pumps.dispense_liquid(volume = 3,
                                           inlet_channel = "air",
                                           outlet_channel = f"cell{i}",
                                           in_speed = in_speed,
                                           out_speed = 200)

        time.sleep(2*60)
        self.bopai_controller.empty_the_cell(output = "electrodeposition_cell_peristaltic_pump", time_sleep = 20)

        # add water to the cells and wash the cells four times.
        for num in range(repeat_num):
            for i in range(batch_num):
                self.pumps.dispense_liquid(volume = reagent_volume,
                                           inlet_channel = reagent_name,
                                           outlet_channel = f"cell{i}",
                                           in_speed = in_speed,
                                           out_speed = out_speed)
            self.bopai_controller.empty_the_cell(output = "electrodeposition_cell_peristaltic_pump", time_sleep = 20)

        for i in range(batch_num):
            self.pumps.dispense_liquid(volume = reagent_volume,
                                       inlet_channel = reagent_name,
                                       outlet_channel = f"cell{i}",
                                       in_speed = in_speed,
                                       out_speed = out_speed)
        for i in range(batch_num):
            for repeat in range(3):
                self.pumps.dispense_liquid(volume = 3,
                                           inlet_channel = "air",
                                           outlet_channel = f"cell{i}",
                                           in_speed = in_speed,
                                           out_speed = out_speed)
        
        
    def refill_cat_cell(self):
        """
        Clean and refill the catalysis cell.
        """
        for i in range (5):
            self.pumps.dispense_liquid(volume = 4,
                                   inlet_channel = "air_2",
                                   outlet_channel = "cell_cat3",
                                   in_speed = 800,
                                   out_speed = 300)

        # clean the cell 
        self.bopai_controller.empty_the_cell(output = "cell_cat3_peristaltic_pump")


        # self.pumps.dispense_liquid(volume = 1,
        #                            inlet_channel = "H2SO4_2",
        #                            outlet_channel = "cell_cat3",
        #                            in_speed = 800,
        #                            out_speed = 300)
        # self.bopai_controller.empty_the_cell(output = "cell_cat3_peristaltic_pump",time_sleep = 5)
        
        # for i in range (3):
        #     self.pumps.dispense_liquid(volume = 3,
        #                            inlet_channel = "air_2",
        #                            outlet_channel = "cell_cat3",
        #                            in_speed = 800,
        #                            out_speed = 300)
            
        # self.bopai_controller.empty_the_cell(output = "cell_cat3_peristaltic_pump", time_sleep = 5)

        # self.pumps.dispense_liquid(volume = 2,
        #                             inlet_channel = "KOH",
        #                             outlet_channel = "cell_cat3",
        #                             in_speed = 800,
        #                             out_speed = 800)
        # self.bopai_controller.empty_the_cell(output = "cell_cat3_peristaltic_pump", time_sleep = 5)
        
        # for i in range(3):
        #     self.pumps.dispense_liquid(volume = 7,
        #                             inlet_channel = "H2O",
        #                             outlet_channel = "cell_cat3",
        #                             in_speed = 800,
        #                             out_speed = 800)
        #     self.bopai_controller.empty_the_cell(output = "cell_cat3_peristaltic_pump")
            
        # self.pumps.dispense_liquid(volume = 7,
        #                             inlet_channel = "KOH",
        #                             outlet_channel = "cell_cat3",
        #                             in_speed = 800,
        #                             out_speed = 800)
        # for i in range (3):
        #     self.pumps.dispense_liquid(volume = 3,
        #                            inlet_channel = "air_2",
        #                            outlet_channel = "cell_cat3",
        #                            in_speed = 800,
        #                            out_speed = 300)
        self.pumps.dispense_liquid(volume = 1,
                                    inlet_channel = "H2O_D2O",
                                    outlet_channel = "cell_cat3",
                                    in_speed = 800,
                                    out_speed = 800)
        self.bopai_controller.empty_the_cell(output="cell_cat3_peristaltic_pump",time_sleep=5)
        self.pumps.dispense_liquid(volume = 7.5,
                                    inlet_channel = "H2O_D2O",
                                    outlet_channel = "cell_cat3",
                                    in_speed = 800,
                                    out_speed = 300)
        for i in range (3):
            self.pumps.dispense_liquid(volume = 3,
                                   inlet_channel = "air_2",
                                   outlet_channel = "cell_cat3",
                                   in_speed = 800,
                                   out_speed = 300)


    def Pt_test(self):
        """
        Testing of Pt to ensure experimental accuracy
        
        """
        self.mass_flag_lock.acquire()
        self.mass_in_progress = True
        self.mass_flag_lock.release()

        # perform electrochemical test  
        self.robotic_platform.led_to_HD_test_cell1(lock = self.arm_lock,
                                                  dry_coil_address = 78)
        Pt_start =  datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.bopai_controller.cat_cell1(output="cat_cell")
        f1 = threading.Thread(target = self.Et_test, kwargs = {"channel":self.channel7, "current":-50,  "test_time":1600} )
        f2  = threading.Thread(target=self.MS_test, )

        f1.start()
        f2.start()

        f1.join()
        f2.join()

        self.channel7.save_result(exp_index = "Pt1", 
                                  save_path = f"./tmp/electrocatalysis/Et/")
        
        # process the mass data and get the separation_coefficient
        self.mass.save_file(lock = self.screen_lock)

        # self.robotic_platform.HD_cell1_to_cell2(lock = self.arm_lock)
        # self.bopai_controller.cat_cell2(output="cat_cell")
        # f1 = threading.Thread(target = self.Et_test, kwargs = {"channel":self.channel8, "current":-50,  "test_time":1600} )
        # f2  = threading.Thread(target=self.MS_test, )
        # f1.start()
        # f2.start()
        # f1.join()
        # f2.join()
        # self.channel8.save_result(exp_index = "Pt2", 
        #                           save_path = f"./tmp/electrocatalysis/Et/")
        
        # # process the mass data and get the separation_coefficient
        # self.mass.save_file(lock = self.screen_lock)
        # self.bopai_controller.cat_cell1(output="cat_cell")
        self.robotic_platform.HD_cell1_to_waste(lock = self.arm_lock)

        self.mass_flag_lock.acquire()
        self.mass_in_progress = False
        self.mass_flag_lock.release()


        # H2_intercept, H2_slope, HD_intercept, HD_slope = process_mass_standard_curve(file_path = self.mass_save_path,
        #                                                                              t_start = self.t_start, 
        #                                                                              t_end = self.t_end)
        # separation_coefficient = process_mass_data(file_path = self.mass_save_path, 
        #                       t_start = Pt_start,
        #                       t_end = Pt_end,
        #                       H2_intercept = H2_intercept, 
        #                       H2_slope = H2_slope,
        #                       HD_intercept = HD_intercept,
        #                       HD_slope = HD_slope)

        # # process the lsv data and get the potential
        # over_potential_current = [-0.010, -0.020, -0.030, -0.040, -0.050]
        # potential = [process_lsv_data(exp_index = "Pt", 
        #                               save_path = f"./tmp/electrocatalysis/Et/", 
        #                               mediana=i) for i in over_potential_current]
        # self.logger.info(f"Pt:separation_coefficient:{separation_coefficient}; potential:{potential[0]}")





