import ctypes
import platform
import os 

# 运行环境判断

systype = platform.system()
if systype == 'Windows':
    if platform.architecture()[0] == '64bit':
        zauxdll = ctypes.WinDLL(
            r'C:\Users\Admin\Documents\GitHub\ec_tool\EC_tool\manager\robotic_arm\motion_controller\dlls\zauxdll.dll')
        print('Windows x64')
    else:
        zauxdll = ctypes.WinDLL(
            r'C:\Users\Admin\Documents\GitHub\ec_tool\EC_tool\manager\robotic_arm\motion_controller\dlls\zauxdll.dllim')
        print('Windows x86')
elif systype == 'Darwin':
    zmcdll = ctypes.CDLL('./dlls/zmotion.dylib')
    print("macOS")
elif systype == 'Linux':
    zmcdll = ctypes.CDLL('./dlls/zmotion.so')
    print("Linux")
else:
    print("OS Not Supported!!")


class ZAUXDLL:
    def __init__(self):
        self.handle = ctypes.c_void_p()

    def ZAux_OpenEth(self, ipaddr):
        '''
        :Description:与控制器建立链接。

        :param ipaddress:IP地址,字符串的方式输入。  

        :Return:错误码。                            

       '''
        ip_bytes = ipaddr.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        ret = zauxdll.ZAux_OpenEth(p_ip, ctypes.pointer(self.handle))
        return ret

    def ZAux_Close(self):
        '''
        :Description:关闭控制器链接。

        :Return:错误码。

       '''
        ret = zauxdll.ZAux_Close(self.handle)
        return ret

    def ZAux_Direct_SetOp(self, ionum, iValue):
        '''
        :Description:操作输出信号。

        :param ionum:输出口编号。  

        :param iValue:输出口状态。 

        :Return:错误码。 

       '''

        ret = zauxdll.ZAux_Direct_SetOp(self.handle, ionum, iValue)
        return ret

    def ZAux_Direct_GetIn(self, ionum):
        '''
        :Description:读取输入信号。

        :param ionum:输入口编号。

        :param iValue:输入口状态。

        :Return:错误码。

       '''

        value = ctypes.c_uint32()
        ret = zauxdll.ZAux_Direct_GetIn(self.handle, ionum, ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetAtype(self, iaxis, iValue):
        '''
        :Description:设置轴类型。

        :param iaxis:轴号。

        :param iValue:轴类型。   

        :Return:错误码。 
        '''

        ret = zauxdll.ZAux_Direct_SetAtype(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_SetAccel(self, iaxis, fValue):
        '''
        :Description:设置加速度。

        :param iaxis:轴号 

        :param fValue:设定值   

        :Return:错误码。 

        '''
        ret = zauxdll.ZAux_Direct_SetAccel(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetDpos(self, iaxis, fValue):
        '''
        :Description:设置轴位置。

        :param iaxis:轴号。        

        :param fValue:设置的坐标值。  

        :Return:错误码。  

        '''

        ret = zauxdll.ZAux_Direct_SetDpos(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetMpos(self, iaxis, fValue):
        '''
        :Description:设置轴位置。

        :param iaxis:轴号。

        :param fValue:设置的坐标值。

        :Return:错误码。

        '''

        ret = zauxdll.ZAux_Direct_SetMpos(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetUnits(self, iaxis, fValue):
        '''
        :Description:设置 脉冲当量。

        :param iaxis:轴号。       

        :param fValue:设置的当量值。     

        :Return:错误码。 

        '''

        ret = zauxdll.ZAux_Direct_SetUnits(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetDecel(self, iaxis, fValue):
        '''
        :Description:设置减速度。

        :param iaxis:轴号。  

        :param fValue:设置的减速度值。 

        :Return:错误码。 
        '''

        ret = zauxdll.ZAux_Direct_SetDecel(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetSpeed(self, iaxis, fValue):
        '''
        :Description:设置轴速度,单位为units/s,当多轴运动时,作为插补运动的速度。

        :param iaxis:轴号。     

        :param fValue:设置的速度值。      

        :Return:错误码。 

        '''

        ret = zauxdll.ZAux_Direct_SetSpeed(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret


    def ZAux_Direct_SetSramp(self, iaxis, fValue):
        '''
        :Description:设置 S曲线设置。 0-梯形加减速。

        :param iaxis:轴号。       

        :param fValue:S曲线平滑时间MS。      

        :Return:错误码。 

        '''
        ret = zauxdll.ZAux_Direct_SetSramp(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetDatumIn(self, iaxis, iValue):
        '''
        :Description:设置原点信号   设定-1为取消原点设置。

        :param iaxis:轴号。  

        :param iValue: 设置的原点信号输入口编号。  

        :Return:错误码。  

        '''

        ret = zauxdll.ZAux_Direct_SetDatumIn(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_SetCreep(self, iaxis, fValue):
        '''
        :Description:设置回零爬行速度。

        :param iaxis:轴号。       

        :param fValue: 设置的速度值。

        :Return:错误码。 

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_SetCreep(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue),
                                           ctypes.byref(value))
        return ret

    def ZAux_Direct_Single_Datum(self, iaxis, imode):
        '''
        :Description:控制器方式回零。

        :param iaxis:轴号。

        :param imode:模式。  

        :Return:错误码。

        '''

        ret = zauxdll.ZAux_Direct_Single_Datum(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_GetHomeStatus(self, iaxis):
        '''
        :Description:回零完成状态。

        :param iaxis:轴号。  

        :Return:错误码,回零完成标志 0-回零异常 1回零成功。

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetHomeStatus(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetIfIdle(self, iaxis):
        '''
        :Description:读取轴是否运动结束。

        :param iaxis:轴号。        

        :Return:错误码,返回运行状态 0-运动中 -1 停止。 

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetIfIdle(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetDpos(self, iaxis):
        '''
        :Description: 读取轴当前位置或称控制器发送的需求位置，单位 units。

        :param iaxis:轴号。

        :Return:错误码,返回运行状态 0-运动中 -1 停止。

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetDpos(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        print(ret, value)
        return ret, value

    def ZAux_Direct_GetSpeed(self, iaxis):
        '''
        :Description:读取轴速度。

        :param iaxis:轴号。

        :Return:错误码,返回运行状态 0-运动中 -1 停止。

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetSpeed(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetParam(self, sParam, iaxis):
        '''
        :Description:通参数 通用的参数读取函数, sParam:填写参数名称。

        :param sParam:轴参数名称 "DPOS" ...   

        :param iaxis:轴号   

        :Return:错误码,读取的返回值。,float

        '''
        value = ctypes.c_float()
        _str = sParam.encode('utf-8')
        ret = zauxdll.ZAux_Direct_GetParam(
            self.handle, _str, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_Move(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:多轴相对直线插补  20130901 以后的控制器版本支持。

        :param imaxaxises:参与轴数。 

        :param piAxislist:轴列表。 

        :param pfDisancelist:距离列表。  

        :Return:错误码。

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (
            ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(
            self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret

    def ZAux_Direct_SetInvertStep(self, iaxis, iValue):
        '''
        :Description:设置脉冲输出模式。

        :param iaxis:轴号。        

        :param iValue:设定的脉冲输出模式 脉冲+方向/双脉冲。        

        :Return:错误码。 
        '''

        ret = zauxdll.ZAux_Direct_SetInvertStep(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_SetParam(self, sParam, iaxis, fset):
        '''
        :Description:通用的参数修改函数 sParam: 填写参数名称。

        :param sParam:轴参数名称 "DPOS" ...   

        :param iaxis:轴号   

        :param fset:设定值  

        :Return:错误码。

        '''
        _str = sParam.encode('utf-8')
        ret = zauxdll.ZAux_Direct_SetParam(
            self.handle, _str, ctypes.c_int(iaxis), ctypes.c_float(fset))
        return ret

    def ZAux_Direct_Single_MoveAbs(self, iaxis, fdistance):
        '''
        :Description:单轴绝对运动。

        :param iaxis:轴号。  

        :param fdistance:距离。   

        :Return:错误码。

        '''

        ret = zauxdll.ZAux_Direct_Single_MoveAbs(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fdistance))
        return ret

    def ZAux_Direct_MoveAbs(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:绝对多轴直线插补  20130901 以后的控制器版本支持。

        :param imaxaxises:参与轴数。 

        :param piAxislist:轴列表。 

        :param pfDisancelist:距离列表。  

        :Return:错误码。

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (
            ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(
            self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret
