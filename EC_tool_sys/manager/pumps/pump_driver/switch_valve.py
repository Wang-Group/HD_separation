from .base import BaseDevice
import serial
import logging
import time

class SwitchValve(BaseDevice):

    def __init__(self, slave, ser: serial.Serial, logger: logging.Logger, commands):
        """
        Initialize the multi-port valve.
        
        Args:
            slave (int): Device address number.
            ser (serial.Serial): Serial connection.
            logger (logging.Logger): Logger for logging information.
            commands (dict): Dictionary of commands.
        """
        super().__init__(slave, ser, logger, commands)
        self.set_commands_dict('SwitchValve')

    def restoration(self, wait_or_not=True):
        """
        Reset the valve to the home position.
        
        Args:
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        self.serial_write('复位', 0x00, 0x00)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def reset_to_origin(self, wait_or_not=True):
        """
        Reset the valve to the origin position.
        
        Args:
            wait_or_not (bool): Flag to indicate if feedback is required.
        """
        self.serial_write('原点复位', 0x00, 0x00)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()
