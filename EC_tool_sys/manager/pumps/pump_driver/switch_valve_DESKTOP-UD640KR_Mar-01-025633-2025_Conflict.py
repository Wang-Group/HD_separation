from .base import BaseDevice
import serial
import logging
import time

class SwitchValve(BaseDevice):

    def __init__(self, slave, ser: serial.Serial, logger: logging.Logger, commands):
        """
        多通阀初始化
        :param slave: 设备地址号
        """
        super().__init__(slave, ser, logger, commands)
        self.set_commands_dict('SwitchValve')

    def restoration(self, wait_or_not=True):
        """
        复位：B3=0x00 B4=0x00切换阀运行到复位光耦处停止
        """
        self.serial_write('复位', 0x00, 0x00)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def reset_to_origin(self, wait_or_not=True):
        """
        原点复位：B3=0x00 B4=0x00 切换阀运行到编码器原点位置，与 0x45 命令复位的位置重叠
        """
        self.serial_write('原点复位', 0x00, 0x00)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()
