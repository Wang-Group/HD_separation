from .zauxdllPython import *
import time
import logging

class MyError(Exception):
    def __init__(self, message, code=9999):
        super().__init__(message)
        self.code = code
    def __str__(self):
        return f'{self.code}: {self.args[0]}'


class MotionController:
    def __init__(self, param, logger: logging.Logger):
        
        # 初始化运动控制器
        self.ipaddr = param['controller_ipaddr']
        self.controller_axis = param['controller_axis']
        self.param_x = param['param_x']
        self.param_datum_x = param['param_datum_x']
        self.op1 = param['op1']
        self.op2 = param['op2']
        self.ip2 = param['ip2']

        self.zaux = ZAUXDLL()
        self.logger = logger
        try:
            ret = self.zaux.ZAux_OpenEth(self.ipaddr)
            if ret != 0:
                raise MyError('fail to connect to the ethernet', ret)
        except MyError as e:
            # 处理自定义异常
            self.logger.error(f'error: {e}, code = {e.code}')
            raise e

        try:
            ret = self.zaux.ZAux_Direct_SetOp(self.op1, 1)
            if ret != 0:
                raise MyError("output error", code=ret)
        except MyError as e:
            # 处理自定义异常
            self.logger.error(f'error: {e}, code = {e.code}')
            raise e
        
        self.home()
        self.set_motion_params()

    def check_command(self, ret: int):
        try:
            if ret != 0:
                raise MyError("the command is not executed, with return {}".format(ret))
        except MyError as e:
            self.logger.error("error:", e)
            raise e

    def home(self):
        # x轴回零
        ret = self.zaux.ZAux_Direct_SetOp(self.op2, 0)
        time.sleep(0.2)
        self.zaux.ZAux_Direct_SetOp(self.op2, 1)

        # while True:
        #     ret, idle = self.zaux.ZAux_Direct_GetIfIdle(self.controller_axis)
        #     if int(idle.value) == -1:
        #         break
        # while True:
        #     ret = self.zaux.ZAux_Direct_GetIn(self.ip2)
        #     # print(ret[1], type(ret[1]))
        #     try:
        #         if ret[1].value == 1:
        #             i = i + 1
        #             if i >= 1000:
        #                 print('回零点成功')
        #                 break
        #         else:
        #             i = 0
        #     except:
        #         pass
        time.sleep(10.5)
        self.logger.info('the axis is homed')
        self.reset_ports()

    def reset_ports(self):
        # 关闭回零输出口
        self.zaux.ZAux_Direct_SetOp(self.op2, 0)
        time.sleep(0.1)
        # 回零成功之后清零X的Mpos和Dpos
        self.zaux.ZAux_Direct_SetDpos(self.controller_axis, 0)
        time.sleep(0.1)
        self.zaux.ZAux_Direct_SetMpos(self.controller_axis, 0)
        time.sleep(0.1)
        self.zaux.ZAux_Direct_GetDpos(self.controller_axis)
        time.sleep(0.1)

    def set_speed(self, speed: float):
        self.zaux.ZAux_Direct_SetSpeed(self.controller_axis, speed)
        while True:
            ret, val = self.zaux.ZAux_Direct_GetSpeed(self.controller_axis)
            if speed == val.value:
                self.logger.error(f"set {self.controller_axis} with a speed of {speed}")
                break

    def set_motion_params(self):
        # 设置运动参数
        self.zaux.ZAux_Direct_SetAtype(self.controller_axis, int(self.param_x[0]))
        time.sleep(0.1)
        self.zaux.ZAux_Direct_SetInvertStep(self.controller_axis, int(self.param_x[1]))
        time.sleep(0.1)
        self.zaux.ZAux_Direct_SetUnits(self.controller_axis, self.param_x[2])
        time.sleep(0.1)
        self.zaux.ZAux_Direct_SetAccel(self.controller_axis, self.param_x[3])
        time.sleep(0.1)
        self.zaux.ZAux_Direct_SetDecel(self.controller_axis, self.param_x[4])
        time.sleep(0.1)
        self.zaux.ZAux_Direct_SetSpeed(self.controller_axis, self.param_x[5])
        time.sleep(0.1)

    def single_move_abs(self, position: float):
        # 单轴绝对运动
        ret = self.zaux.ZAux_Direct_Single_MoveAbs(
            self.controller_axis, position)
        self.check_command(ret)
        t1 = time.time()
        while True:
            ret, idle = self.zaux.ZAux_Direct_GetIfIdle(self.controller_axis)
            if int(idle.value) == -1:
                break

        t2 = time.time()
        while (t2-t1) < (10.5/620*position):
            t2 = time.time()
            time.sleep(0.1)

        try:
            ret, mpos = self.zaux.ZAux_Direct_GetDpos(self.controller_axis)
            mpos = round(float(mpos.value), 2)
            if mpos != position:
                raise MyError(f'{self.controller_axis} does not move to the position', ret)
        except Exception as e:
            # 处理其他异常
            self.logger.error('error：', e)
            # 抛出异常停止程序继续往下执行
            raise e
