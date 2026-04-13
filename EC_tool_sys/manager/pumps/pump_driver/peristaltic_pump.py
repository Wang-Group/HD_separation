from .base import BaseDevice
import serial
import logging


class PeristalticPump(BaseDevice):

    def __init__(self, slave, ser: serial.Serial, logger: logging.Logger, commands):
        """
        :param slave: Peristaltic pump address
        """

        super().__init__(slave, ser, logger, commands)
        self.set_commands_dict('PeristalticPump')
        self.speed = 100

    def forced_to_stop(self):
        """
        Force stop.
        """
        self.ser.write(self.make_frame('强停', 0X00, 0X00))
        return self.ser.read(8)

    def continue_clockwise_motion(self, wait_or_not=True):
        """
        Run continuously clockwise: B3=0x00, B4=0x00.
        """
        self.ser.write(self.make_frame('持续顺时针运行', 0x00, 0x00))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def continue_counterclockwise_motion(self, wait_or_not=True):
        """
        Run continuously counterclockwise: B3=0x00, B4=0x00.
        """
        self.ser.write(self.make_frame('持续逆时针运行', 0x00, 0x00))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def step_clockwise_motion(self, step_num, wait_or_not=True):
        """
        step_num: number of steps for stepped clockwise motion
        """
        step_lsb = step_num & 0xFF
        step_msb = (step_num >> 8) & 0xFF
        self.ser.write(self.make_frame('按步顺时针运行', step_lsb, step_msb))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def step_counterclockwise_motion(self, step_num, wait_or_not=True):
        """
        step_num: number of steps for stepped counterclockwise motion
        """
        step_lsb = step_num & 0xFF
        step_msb = (step_num >> 8) & 0xFF
        self.ser.write(self.make_frame('按步逆时针运行', step_lsb, step_msb))
        self.ser.read(8)
        if wait_or_not:
            self.feedback()

    def set_dynamic_speed(self, speed):
        """
        Set the motor rotation speed.
        :param speed: Speed in rpm; the default after power-up is 100 rpm
        """
        speed_lsb = speed & 0xFF
        speed_msb = (speed >> 8) & 0xFF
        self.ser.write(self.make_frame('设置动态转速', speed_lsb, speed_msb))
        self.ser.read(8)
        self.speed = speed

