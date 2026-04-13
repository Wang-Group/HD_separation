from .ascii_commands import COMMANDS
from serial import Serial
from logging import Logger
import time
from typing import Optional, Union
MODEL = {
        "SY-06B": {"max_step": 6000, 
                    "max_distance": 60, 
                    "lead": 2},
        "SY-08": {
                "max_step": 12000, 
                "max_distance": 30, 
                "lead": 1}, 
        "SY-03B" : {
                "max_step": 6000, 
                "max_distance": 60, 
                "lead": 6},

        "SY-09S" : {"max_step": 7680, 
                    "max_distance": 19.2, 
                    "lead": 1},

        "SY-09S-3mL" : {
            "max_step": 7200, 
            "max_distance": 18, 
            "lead": 1
            }

}
STEP_MODE={
    "SY-06B":{
        0: 1, 
        1: 8, 
        2: 8},
    "SY-09S":{
        0:1,
        1:8,
        2:8},
    "SY-09S-3mL":{
        0:1,
        1:8,
        2:8}
}

class SyringePump_ASCII:
    def __init__(self, ser: Serial, slave: int, model: str, volume: float, logger: Optional[Logger] = None, feedback0: Optional[bool] = True) -> None:
        """
        initialization the parameters and force the pump reset
        Args: ser: Serial(port = "COMx", baudrate = 9600/115200), it is an instance of the class--- Serial 
              slave: the address of the pump, it distributes from 0 to 14.
              model: the model of the pump.
              volume: the volume of the pump, the unit is ml
              logger: the logging of the class
              feedback0: if you don't need the feedback of the pump, you can set it to "False"

        
        """
        self.ser = ser
        self.slave = slave
        self.model = model
        self.max_step = MODEL[model]["max_step"]
        self.max_distance = MODEL[model]["max_distance"]
        self.lead = MODEL[model]["lead"]
        self.commands = COMMANDS
        self.volume = volume
        self.logger = logger
        self.valve_port = {0: "switch_to_input_channel", 1: "switch_to_output_channel"}
        self.valve_feedback = {"SY-06B":{0: "i", 1: "o",},
                               "SY-09S":{0: "i", 1: "o"},
                               "SY-09S-3mL":{0: "i", 1: "o"}
                               }
        self.step_mode = 0
        self.feedback0 = feedback0
        self.STEP_MODE = STEP_MODE[self.model]
    
        self.force_reset(feedback = self.feedback0)

    def force_reset(self, feedback : bool = True):
        """
        force the pump reset
        """
        self.send_message("initialize_pump&valve_CW")
        if feedback:
            self.feedback("read_absolute_position",0)

    def generate(self, func: str, para: Optional[int] = None):
        """
        generate the command sending to the pump

        Args: func: the function you want to run
              para: when you  use some function, you need to input the value, such as the step, step / second, input channel or the output channel  
        """
        STX = self.commands["STX"]
        ETX = self.commands["ETX"]
        SLAVE = str(self.slave + 1)
        FUNC = self.commands[func]
        if para != None:
            PARA = str(para)
        else:
            PARA = ""
        NL = "\r"
        MSG = (STX + SLAVE + FUNC +PARA + ETX + NL).encode('utf-8')
        return MSG

    def send_message(self, func: str, para: Optional[int] = None, timeout: int = 20): 
        """"
        send the command to the pump

        Args: func: the function you want to run
              para: when you  use some function, you need to input the value, such as the step, step / second, input channel or the output channel 
        """
        start_time = time.time()
        self.ser.read_all()
        self.ser.write(self.generate(func,para))
        while True:
            time.sleep(0.1)
            receive_message = self.ser.read_all()
            if receive_message != b'':
                break
            self.ser.write(self.generate(func,para))
            if time.time() -start_time >= timeout:
                raise TimeoutError(f"feedback timeout, message: {self.generate(func, para)}")
        return receive_message

    def feedback(self, func: str, wanted: Optional[int] = None, timeout: Optional[int] = 6000, TIME_OUT: Optional[bool] = True): 
        """
        judge if the device reach the goal, the code does not continue until the device reach the goal or raise error 

        Args: func: the function you want to inquire
              wanted: the goal you set 
              timeout: set the max time for the loop. if the running time is lagger than that, it will raise the TIMEOUT error. the default value si 10 second
              TIME_OUT: if the running time is quite big and you want run the loop until the device reach the goal, you can set the para to "False", the  default value is "True"
        """
        start_time = time.time()
        while True:
            feedback_value = self.send_message(func)
            if func == "read_device_status" :
                if feedback_value[3:4].decode() == "`" :
                    return True              #the device is off
            if func == "read_absolute_position":
                try:
                    if feedback_value[3:4].decode() == "`": #the device is off
                        current_position = int(str(feedback_value).split("`")[-1].split("\\")[0])
                        current_volume = current_position / self.max_step * self.volume
                        wanted_volume =  wanted/ self.max_step * self.volume
                        if current_position - wanted in [-1,0,1] : 
                            self.logger.info(f"absolute move({wanted_volume} ml) succeed")
                            return True
                        else:
                            time.sleep(0.1)
                            self.logger.error(f"the position is not so correct, set volume is {wanted_volume} mL, current position is {current_volume} mL")
                    else:
                        self.logger.warning("the readout format is not correct. Skiping the current feedback.")
                except:
                    pass
            if func == "read_valve_position" and "`" in str(feedback_value): 
                current_valve_position = str(feedback_value).split("`")[1].split("\\")[0]
                time.sleep(0.1)
                if current_valve_position == self.valve_feedback[self.model][wanted]: 
                    self.logger.info(f"succeed in opening {wanted} valve ")
                    return True
                else:
                    self.logger.error(f"failed in opening {wanted} valve")
            if func == "read_step_mode" and "`" in str(feedback_value): 
                current_step_mode = str(feedback_value).split("`")[1].split("\\")[0]
                if current_step_mode == str(wanted):
                    return True
            if func == "read_max_speed":
                current_speed = int(str(feedback_value).split("`")[1].split("\\")[0])
                if current_speed == wanted:
                    self.logger.info(f"set_dynamic_speed succeed, the speed is {wanted}")
                    return True
                else:
                    self.logger.error(f"failed in seting speed, the speed is {current_speed}")
                    time.sleep(0.1)
            if TIME_OUT:
                if time.time() - start_time >= timeout:
                    raise TimeoutError(f'feedback timeout, error code:{feedback_value}')
                    
                time.sleep(0.1)

    def set_step_mode(self, mode: int, feedback: bool = True):
        """
        set the step mode of the pump

        Args: mode: 0 is normal mode, 1 is fine-tuning the positioning model, 2 is niche model.
              feedback:if you don't need the feedback, you can set it to "False". the  default value is "True".
        """
        self.send_message("step_mode",mode)
        if feedback:
            self.feedback(func = "read_step_mode",wanted = mode)
        self.step_mode = mode
        self.logger.info(f"succeed in seting {mode} step mode")


    def move_to_absolute_position(self, absolute_volume: float, feedback: bool = False, timeout: float = 3000): 
        """
        move the pump to the absolute positision

        Args: absolute_volume: the volume of liquid you want to pump, the unit is ml
              feedback: if you don't need the feedback, you can set it to "False". the  default value is "True".
              timeout : the feedback timeout
        """
        if absolute_volume > self.volume:
            self.logger.error(f"{self.slave} pump: absolute_volume should not bigger than the max volume.")
            raise ValueError(f"{self.slave} pump: absolute_volume should not bigger than the max volume.")
        step = round(absolute_volume * self.max_step / self.volume)*STEP_MODE[self.model][self.step_mode]
        self.send_message("absolute_move",step)
        time.sleep(0.1)
        if feedback:
            self.feedback("read_absolute_position", step, timeout = timeout)

    def set_dynamic_speed(self, speed: float = 1.75, feedback: Optional[bool] = True):
        """
        Set the dynamic speed of the pump.

        Args:
            speed: the speed of motor, unit: mL/s. Defaults to 1.75.
            feedback: if you don't need the feedback, you can set it to "False". the  default value is "True".
        """
        volume_speed = round(speed / self.volume * self.max_step * STEP_MODE[self.model][self.step_mode]) 
        self.send_message("set_max_speed", volume_speed)
        time.sleep(0.1)
        if feedback:
            self.feedback("read_max_speed", volume_speed)


    def switch_valve(self, port: int, feedback: Optional[bool] = True): 
        """
        Switch the valve of the pump
        
        Args:
            port: the channel you want to switch to
                for "SY-06B": 1: "switch_to_input_channel", 0: "switch_to_output_channel"
                for "SY-09S": 0: "NO", 1: "NC"        
        """
        if self.model == "SY-06B":
            self.send_message(func = self.valve_port[port])
            time.sleep(0.1)
            if feedback:
                self.feedback("read_valve_position", port)
        if self.model == "SY-09S":
            self.send_message("MOS_switch_valve", port)
            time.sleep(0.1)
            if feedback:
                self.feedback("read_valve_position", port)
    def rinse_the_pump(self, volume: Optional[float] = 4, feedback: Optional[bool] = True):
        """
        rinse the pump

        Args: volume: the volume of the pump you want to rinse. the default value is 4
              feedback: if you don't need the feedback, you can set it to "False". the  default value is "True".
        """
        for i in range(2): 
            self.switch_valve(0)
            self.move_to_absolute_position(volume)
            self.switch_valve(1)
            self.move_to_absolute_position(0)

    def transfer_liquid(self, volume: float, in_speed: Optional[float] = 0.25, out_speed: float = 1.75, inlet: Union[str, int] = None, outlet: Union[str, int] = None):
        """
        transfer designated volume liquid at designated speed, in designated direction
        Args: volume: the volume of liquid you want to transfer, unit is ml
              speed: the speed of transfering the liquid, unit is ml/s
              inlet:
              outlet:
            timeout: to avoid the runtime of the pump is lagger than the default of timeout of the feedback function, timeout is (volume / speed + 60)
        
        """
        timeout = volume/in_speed + 60.0
        while volume > 0:
            self.switch_valve(inlet)
            if volume < self.volume or volume == self.volume:
                volume_to_go = volume
                volume = 0
            else:
                volume_to_go = self.volume
                volume = volume - self.volume
            self.set_dynamic_speed(speed = in_speed)
            self.move_to_absolute_position(absolute_volume=volume_to_go, timeout=timeout)
            time.sleep(1)
            self.switch_valve(outlet)
            self.set_dynamic_speed(speed=out_speed)
            self.move_to_absolute_position(absolute_volume=0, timeout= timeout)
            # turn the valve off
            self.switch_valve(0)

    

    def ensure_move_to_absolute_position(self, absolute_volume: float, timeout: float = 3000): 
        """
        move the pump to the absolute positision

        Args: absolute_volume: the volume of liquid you want to pump, the unit is ml
              feedback: if you don't need the feedback, you can set it to "False". the  default value is "True".
              timeout : the feedback timeout
        """
        step = round(absolute_volume * self.max_step / self.volume)*STEP_MODE[self.model][self.step_mode]
        time.sleep(1)
        self.feedback("read_absolute_position", step, timeout = timeout)
        time.sleep(1)
