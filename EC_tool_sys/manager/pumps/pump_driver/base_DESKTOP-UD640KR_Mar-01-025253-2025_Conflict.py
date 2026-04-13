import time
import serial
import logging


def checksum(lst):

    """

    输入一个列表，计算2字节的16进制校验和，并按低位在前输出一个列表
    :param lst: 输入列表
    :return: 输出校验和列表
    """
    checksum_hex = hex(sum(lst)).lstrip('0x')
    checksum_str = '{:0>4s}'.format(checksum_hex)
    checksum_lst = [int(checksum_str[2:], 16), int(checksum_str[0:2], 16)]
    return checksum_lst


def read_file(file):
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
        设备初始化
        :param slave: 设备地址号
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
            if installation == 'SwitchValve':
                self.commands = {**self.Common_commands, **self.SwitchValve_commands}
            elif installation == 'SyringePump':
                self.commands = {**self.Common_commands, **self.SyringePump_commands}
            elif installation == 'PeristalticPump':
                self.commands = {**self.Common_commands, **self.PeristalticPump_commands}

    def make_frame(self, B2: str, B3, B4):
        """
        :param B2: 说明书中功能码
        :param B3: 参数区
        :param B4: 参数区
        :return: bytes(frame)
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
                # self.logger.info('电机正常')
                break
            # time.sleep(0.2)

    def serial_write(self,  B2: str, B3, B4):
        self.ser.write(self.make_frame(B2, B3, B4))

    def valve_switch(self, param, channel: str, feedback_str: str = '当前通道位置', wait_or_not=True):
        """
        阀门切换, 电机通过码盘转动，自动选择最优路径，根据切换阀实际通道数而定
        :param file: 各个阀通道对应的物质的txt文档
        :param substance: 要转换的物质
        :param feedback_str: 各设备查询当前通道位置的名称
        :param wait_or_not: 是否反馈电机状态
        :return: None
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
        按需求方向切换孔位：根据切换阀实际通道数而定，参数值不可超过当前阀的最大通道数，且 B3,B4 必须为相邻的两个孔位
         :param B3:要切换到的通道
         :param B4: 经过B4通道切换到B3（切换方向）
         :param wait_or_not: 是否反馈电机状态
         :return: None
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
        按需求方向切换到孔位之间：根据切换阀实际通道数而定，参数值不可超过当前阀的最大通道数，且 B3,B4 必须为相邻的两个孔位
        :param B3: 阀门号1（B3）
        :param B4: 阀门号2（B4）
        :param wait_or_not: 是否反馈电机状态
        :return: None
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
        强停
        """
        self.serial_write('强停', 0X00, 0X00)
        self.ser.read_all()
    def close(self):
        self.ser.close()
        # self.logger.info('端口已关闭')
