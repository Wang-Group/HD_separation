from .base import BaseDevice
import time
import serial
import logging


class SyringePump(BaseDevice):

    def __init__(self, slave, ser: serial.Serial, logger: logging.Logger, commands):
        """
        注射泵初始化f
        """
        super().__init__(slave, ser, logger, commands)
        self.set_commands_dict('SyringePump')

    def valve_restoration(self, wait_or_not=True):
        """
        阀复位：B3=0x00 B4=0x00切换阀运行到复位光耦处停止
        """
        self.serial_write('阀复位', 0x00, 0x00)
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def pump_restoration(self, wait_or_not=True):
        """
        泵复位：B3=0x00 B4=0x00 切换阀运行到编码器原点位置，与 强制复位的位置重叠
        """
        self.ser.write(self.make_frame('泵复位', 0x00, 0x00))
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def forced_reset(self, wait_or_not=True):
        """
        强制复位：B3=0x00 B4=0x00
        """
        self.ser.write(self.make_frame('强制复位', 0x00, 0x00))
        time.sleep(0.1)
        self.ser.read_all()
        if wait_or_not:
            self.feedback()

    def move_to_absolute_position(self, absolute_volume, measuring_range = 5, wait_or_not=True):
        """
        注射泵运动到绝对位置
        :param absolute_volume: 量程范围：0-5mL   单位：mL
        :param wait_or_not:是否反馈
        :return:none
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
            self.logger.warning("error in checking  the absolute position of the syringe pump ")


    def set_dynamic_speed(self, speed: int):
        """
        设定电机动态转速
        :param speed:电机转动速度：1-900
        :return:none
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