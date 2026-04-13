import time
import serial
import logging


def checksum(lst):
    """
    Calculate 2-byte hexadecimal checksum and return as a list.
    
    Args:
        lst (list): Input list.
    
    Returns:
        list: Checksum list.
    """
    checksum_hex = hex(sum(lst)).lstrip('0x')
    checksum_str = '{:0>4s}'.format(checksum_hex)
    checksum_lst = [int(checksum_str[2:], 16), int(checksum_str[0:2], 16)]
    return checksum_lst


def read_file(file):
    """
    Read parameters from a file.
    
    Args:
        file (str): Path to the file.
    
    Returns:
        dict: Dictionary of parameters.
    """
    with open(file, 'r', encoding='utf-8') as file:
        param = {}
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split(',')
                key = key.strip()
                value = int(value.strip(), 16)
                param[key] = value
    return param


class BaseDevice:

    def __init__(self, slave, ser: serial.Serial, logger: logging.Logger, commands):
        """
        Initialize the device.
        
        Args:
            slave (int): Device address number.
            ser (serial.Serial): Serial connection.
            logger (logging.Logger): Logger for logging information.
            commands (dict): Dictionary of commands.
        """
        self.commands = {}
        self.Common_commands = {}
        self.SwitchValve_commands = {}
        self.SyringePump_commands = {}
        self.PeristalticPump_commands = {}

        self.logger = logger
        self.ser = ser
        self.slave = slave
        self.volume = 0
        self.speed = 0

        self.Common_commands = commands["Common_commands"]
        self.SwitchValve_commands = commands["SwitchValve_commands"]
        self.SyringePump_commands = commands["SyringePump_commands"]
        self.PeristalticPump_commands = commands["PeristalticPump_commands"]

    def set_commands_dict(self, installation='SwitchValve' or 'SyringePump' or 'PeristalticPump'):
        """
        Set the commands dictionary based on the installation type.
        
        Args:
            installation (str): Type of installation ('SwitchValve', 'SyringePump', or 'PeristalticPump').
        """
        if installation == 'SwitchValve':
            self.commands = {**self.Common_commands, **self.SwitchValve_commands}
        elif installation == 'SyringePump':
            self.commands = {**self.Common_commands, **self.SyringePump_commands}
        elif installation == 'PeristalticPump':
            self.commands = {**self.Common_commands, **self.PeristalticPump_commands}

    def make_frame(self, B2: str, B3, B4):
        """
        Create a frame for communication.
        
        Args:
            B2 (str): Function code.
            B3 (int): Parameter.
            B4 (int): Parameter.
        
        Returns:
            bytes: Frame as bytes.
        """
        frame_without_checksum = [self.commands['帧头'],
                                  self.slave,
                                  self.commands[B2],
                                  B3,
                                  B4,
                                  self.commands['帧尾']]

        frame = frame_without_checksum + checksum(frame_without_checksum)
        # print(bytes(frame))
        return bytes(frame)

    def feedback(self):
        """
        Get feedback from the device.
        """
        while True:
            self.ser.write(self.make_frame('电机状态', 0x00, 0x00))
            time.sleep(0.1)
            feedback = self.ser.read_all()
            frame_without_checksum = feedback[:-2]
            checksum_calculated = checksum(frame_without_checksum)
            checksum_received = feedback[-2:]
            # time.sleep(0.2)
            # print(feedback)
            if checksum_received == bytes(checksum_calculated):
                pass
            else:
                self.logger.warning('checksum is not consistent')
                # self.logger.info(str(checksum_received))
                # self.logger.info(str(bytes(checksum_calculated)))

            # time.sleep(0.1)
            if feedback[2] == 0:
                # self.logger.info('Motor is running normally')
                break
            # time.sleep(0.2)

    def serial_write(self,  B2: str, B3, B4):
        """
        Write data to the serial port.
        
        Args:
            B2 (str): Function code.
            B3 (int): Parameter.
            B4 (int): Parameter.
        """
        self.ser.write(self.make_frame(B2, B3, B4))

    def valve_switch(self, param, channel: str, feedback_str: str = '当前通道位置', wait_or_not=True):
        """
        Switch the valve to a specified channel.
        
        Args:
            param (dict): Dictionary of parameters.
            channel (str): Channel to switch to.
            feedback_str (str): Feedback string.
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        self.serial_write('阀切换', int(param[channel], 16), 0x00)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()
        self.serial_write(feedback_str, 0x00, 0x00)
        time.sleep(0.1)
        self.logger.info(f'当前通道位置：{self.ser.read_all()}')

    def valve_switch_by_direction(self, B3, B4, wait_or_not=True):
        """
        Switch the valve to a specified channel by direction.
        
        Args:
            B3 (int): Channel to switch to.
            B4 (int): Intermediate channel.
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        self.serial_write('阀切换带方向', B3, B4)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()
        self.serial_write('当前通道位置', 0x00, 0x00)
        time.sleep(0.1)
        self.logger.info(f'当前通道位置：{self.ser.read_all()}')

    def between_valve_switch_by_direction(self, B3, B4, wait_or_not=True):
        """
        Switch the valve between two specified channels by direction.
        
        Args:
            B3 (int): Channel 1.
            B4 (int): Channel 2.
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        # self.logger.info('between_valve_switch_by_direction')
        self.serial_write('阀切换两孔之间', B3, B4)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()
        self.serial_write('当前通道位置', 0x00, 0x00)
        time.sleep(0.1)
        self.logger.info(f'通道阀的位置：{self.ser.read_all()}')

    def forced_to_stop(self):
        """
        Force the device to stop.
        """
        self.serial_write('强停', 0X00, 0X00)
        self.ser.read_all()

    def close(self):
        """
        Close the serial connection.
        """
        self.ser.close()
   
