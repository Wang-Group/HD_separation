from .base import BaseDevice
import time
import serial
import logging


class SyringePump(BaseDevice):

    def __init__(self, slave, ser: serial.Serial, logger: logging.Logger, commands):
        """
        Initialize the syringe pump.
        
        Args:
            slave (int): Device address number.
            ser (serial.Serial): Serial connection.
            logger (logging.Logger): Logger for logging information.
            commands (dict): Dictionary of commands.
        """
        super().__init__(slave, ser, logger, commands)
        self.set_commands_dict('SyringePump')

    def valve_restoration(self, wait_or_not=True):
        """
        Reset the valve to the home position.
        
        Args:
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        self.serial_write('阀复位', 0x00, 0x00)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def pump_restoration(self, wait_or_not=True):
        """
        Reset the pump to the origin position.
        
        Args:
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        self.ser.write(self.make_frame('泵复位', 0x00, 0x00))
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def forced_reset(self, wait_or_not=True):
        """
        Force reset the pump.
        
        Args:
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        self.ser.write(self.make_frame('强制复位', 0x00, 0x00))
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def move_to_absolute_position(self, absolute_volume, measuring_range = 5, wait_or_not=True):
        """
        Move the syringe pump to an absolute position.
        
        Args:
            absolute_volume (float): Volume range (0-5 mL).
            measuring_range (int): Measuring range.
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        position = round(absolute_volume * 1000 * 3000 / (measuring_range*1000))
        hex_num = hex(position)[2:]
        hex_num = hex_num.zfill(4)
        B3 = int(hex_num[2:], 16)
        B4 = int(hex_num[:2], 16)
        self.ser.write(self.make_frame('注射器到绝对位置', B3, B4))
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()
        try:
            self.ser.write(self.make_frame('查询注射泵位置', 0x00, 0x00))
            time.sleep(0.1)
            a = self.ser.read_all()
            # self.logger.info(str(a))
        except:
            self.logger.warning("error in checking the absolute position of the syringe pump")

    def set_dynamic_speed(self, speed: int):
        """
        Set the dynamic speed of the motor.
        
        Args:
            speed (int): Motor speed (1-900).
        """
        hex_num = hex(speed)[2:]
        hex_num = hex_num.zfill(4)
        B3 = int(hex_num[2:], 16)
        B4 = int(hex_num[:2], 16)
        self.ser.write(self.make_frame('设定动态转速', B3, B4))
        time.sleep(0.1)
        self.ser.read_all()
        self.ser.write(self.make_frame('查询泵运行速度', 0x00, 0x00))
        time.sleep(0.1)
        a = self.ser.read_all()
        self.speed = speed