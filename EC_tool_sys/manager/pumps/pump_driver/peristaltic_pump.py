from .base import BaseDevice
import serial
import logging


class PeristalticPump(BaseDevice):

    def __init__(self, slave, ser: serial.Serial, logger: logging.Logger, commands):
        """
        :param slave: 蠕动泵的地址
        """

        super().__init__(slave, ser, logger, commands)
        self.set_commands_dict('PeristalticPump')
        self.speed = 100

    def forced_to_stop(self):
        """
        强停
        """
        self.ser.write(self.make_frame('强停', 0X00, 0X00))
        return self.ser.read(8)

    def continue_clockwise_motion(self, wait_or_not=True):
        """
        持续顺时针运行：B3=0x00 B4=0x00
        """
        self.ser.write(self.make_frame('持续顺时针运行', 0x00, 0x00))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def continue_counterclockwise_motion(self, wait_or_not=True):
        """
        持续逆时针运行：B3=0x00 B4=0x00
        """
        self.ser.write(self.make_frame('持续逆时针运行', 0x00, 0x00))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def step_clockwise_motion(self, step_num, wait_or_not=True):
        """
        step_num:步数，按步顺时针运行
        """
        step_lsb = step_num & 0xFF
        step_msb = (step_num >> 8) & 0xFF
        self.ser.write(self.make_frame('按步顺时针运行', step_lsb, step_msb))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def step_counterclockwise_motion(self, step_num, wait_or_not=True):
        """
        step_num:步数，按步逆时针运行
        """
        step_lsb = step_num & 0xFF
        step_msb = (step_num >> 8) & 0xFF
        self.ser.write(self.make_frame('按步逆时针运行', step_lsb, step_msb))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def set_dynamic_speed(self, speed):
        """
        设置电机转动速度
        :param speed: 单位为rpm，重新上电时默认速度为100rpm
        """
        speed_lsb = speed & 0xFF
        speed_msb = (speed >> 8) & 0xFF
        self.ser.write(self.make_frame('设置动态转速', speed_lsb, speed_msb))
        self.ser.read(8)
        self.speed = speed

