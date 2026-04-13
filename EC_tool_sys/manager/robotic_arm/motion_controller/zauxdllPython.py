import ctypes
import platform
import os 

# Detect the runtime environment

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
        :Description:Establish a connection to the controller.

        :param ipaddress: IP address provided as a string.

        :Return: Error code.

       '''
        ip_bytes = ipaddr.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        ret = zauxdll.ZAux_OpenEth(p_ip, ctypes.pointer(self.handle))
        return ret

    def ZAux_Close(self):
        '''
        :Description:Close the controller connection.

        :Return: Error code.

       '''
        ret = zauxdll.ZAux_Close(self.handle)
        return ret

    def ZAux_Direct_SetOp(self, ionum, iValue):
        '''
        :Description:Set an output signal.

        :param ionum: Output channel index.

        :param iValue: Output channel state.

        :Return: Error code.

       '''

        ret = zauxdll.ZAux_Direct_SetOp(self.handle, ionum, iValue)
        return ret

    def ZAux_Direct_GetIn(self, ionum):
        '''
        :Description:Read an input signal.

        :param ionum: Input channel index.

        :param iValue: Input channel state.

        :Return: Error code.

       '''

        value = ctypes.c_uint32()
        ret = zauxdll.ZAux_Direct_GetIn(self.handle, ionum, ctypes.byref(value))
        return ret, value

    def ZAux_Direct_SetAtype(self, iaxis, iValue):
        '''
        :Description:Set the axis type.

        :param iaxis: Axis number.

        :param iValue: Axis type.

        :Return: Error code.
        '''

        ret = zauxdll.ZAux_Direct_SetAtype(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_SetAccel(self, iaxis, fValue):
        '''
        :Description:Set acceleration.

        :param iaxis: Axis number.

        :param fValue: Configured value.

        :Return: Error code.

        '''
        ret = zauxdll.ZAux_Direct_SetAccel(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetDpos(self, iaxis, fValue):
        '''
        :Description:Set the axis position.

        :param iaxis: Axis number.

        :param fValue: Position value to set.

        :Return: Error code.

        '''

        ret = zauxdll.ZAux_Direct_SetDpos(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetMpos(self, iaxis, fValue):
        '''
        :Description:Set the machine position of the axis.

        :param iaxis: Axis number.

        :param fValue: Position value to set.

        :Return: Error code.

        '''

        ret = zauxdll.ZAux_Direct_SetMpos(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetUnits(self, iaxis, fValue):
        '''
        :Description:Set the pulse equivalent.

        :param iaxis: Axis number.

        :param fValue: Pulse-equivalent value to set.

        :Return: Error code.

        '''

        ret = zauxdll.ZAux_Direct_SetUnits(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetDecel(self, iaxis, fValue):
        '''
        :Description:Set deceleration.

        :param iaxis: Axis number.

        :param fValue: Deceleration value to set.

        :Return: Error code.
        '''

        ret = zauxdll.ZAux_Direct_SetDecel(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetSpeed(self, iaxis, fValue):
        '''
        :Description:Set axis speed in units/s; for multi-axis motion this is used as the interpolation speed.

        :param iaxis: Axis number.

        :param fValue: Speed value to set.

        :Return: Error code.

        '''

        ret = zauxdll.ZAux_Direct_SetSpeed(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret


    def ZAux_Direct_SetSramp(self, iaxis, fValue):
        '''
        :Description:Set the S-curve profile. A value of 0 means trapezoidal acceleration/deceleration.

        :param iaxis: Axis number.

        :param fValue: S-curve smoothing time in ms.

        :Return: Error code.

        '''
        ret = zauxdll.ZAux_Direct_SetSramp(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue))
        return ret

    def ZAux_Direct_SetDatumIn(self, iaxis, iValue):
        '''
        :Description:Set the home signal. A value of -1 cancels the home-signal setting.

        :param iaxis: Axis number.

        :param iValue: Input channel index for the home signal.

        :Return: Error code.

        '''

        ret = zauxdll.ZAux_Direct_SetDatumIn(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_SetCreep(self, iaxis, fValue):
        '''
        :Description:Set the homing creep speed.

        :param iaxis: Axis number.

        :param fValue: Speed value to set.

        :Return: Error code.

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_SetCreep(self.handle, ctypes.c_int(iaxis), ctypes.c_float(fValue),
                                           ctypes.byref(value))
        return ret

    def ZAux_Direct_Single_Datum(self, iaxis, imode):
        '''
        :Description:Run controller-based homing.

        :param iaxis: Axis number.

        :param imode: Homing mode.

        :Return: Error code.

        '''

        ret = zauxdll.ZAux_Direct_Single_Datum(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(imode))
        return ret

    def ZAux_Direct_GetHomeStatus(self, iaxis):
        '''
        :Description:Get the homing completion status.

        :param iaxis: Axis number.

        :Return: Error code and homing status flag; 0 means homing error and 1 means success.

        '''
        value = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetHomeStatus(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetIfIdle(self, iaxis):
        '''
        :Description:Read whether axis motion has finished.

        :param iaxis: Axis number.

        :Return: Error code and motion state; 0 means moving and -1 means stopped.

        '''
        value = ctypes.c_int()
        ret = zauxdll.ZAux_Direct_GetIfIdle(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetDpos(self, iaxis):
        '''
        :Description:Read the current axis position, or the requested position sent by the controller, in units.

        :param iaxis: Axis number.

        :Return: Error code and position value.

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetDpos(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        print(ret, value)
        return ret, value

    def ZAux_Direct_GetSpeed(self, iaxis):
        '''
        :Description:Read the axis speed.

        :param iaxis: Axis number.

        :Return: Error code and speed value.

        '''
        value = ctypes.c_float()
        ret = zauxdll.ZAux_Direct_GetSpeed(
            self.handle, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_GetParam(self, sParam, iaxis):
        '''
        :Description:Generic parameter-reading function; fill in the parameter name in sParam.

        :param sParam: Axis parameter name, such as "DPOS".

        :param iaxis: Axis number.

        :Return: Error code and the retrieved float value.

        '''
        value = ctypes.c_float()
        _str = sParam.encode('utf-8')
        ret = zauxdll.ZAux_Direct_GetParam(
            self.handle, _str, ctypes.c_int(iaxis), ctypes.byref(value))
        return ret, value

    def ZAux_Direct_Move(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:Multi-axis relative linear interpolation, supported by controller versions after 20130901.

        :param imaxaxises: Number of participating axes.

        :param piAxislist: Axis list.

        :param pfDisancelist: Distance list.

        :Return: Error code.

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (
            ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(
            self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret

    def ZAux_Direct_SetInvertStep(self, iaxis, iValue):
        '''
        :Description:Set the pulse output mode.

        :param iaxis: Axis number.

        :param iValue: Pulse output mode, such as pulse+direction or dual-pulse.

        :Return: Error code.
        '''

        ret = zauxdll.ZAux_Direct_SetInvertStep(
            self.handle, ctypes.c_int(iaxis), ctypes.c_int(iValue))
        return ret

    def ZAux_Direct_SetParam(self, sParam, iaxis, fset):
        '''
        :Description:Generic parameter-setting function; fill in the parameter name in sParam.

        :param sParam: Axis parameter name, such as "DPOS".

        :param iaxis: Axis number.

        :param fset: Value to set.

        :Return: Error code.

        '''
        _str = sParam.encode('utf-8')
        ret = zauxdll.ZAux_Direct_SetParam(
            self.handle, _str, ctypes.c_int(iaxis), ctypes.c_float(fset))
        return ret

    def ZAux_Direct_Single_MoveAbs(self, iaxis, fdistance):
        '''
        :Description:Single-axis absolute motion.

        :param iaxis: Axis number.

        :param fdistance: Distance value.

        :Return: Error code.

        '''

        ret = zauxdll.ZAux_Direct_Single_MoveAbs(
            self.handle, ctypes.c_int(iaxis), ctypes.c_float(fdistance))
        return ret

    def ZAux_Direct_MoveAbs(self, imaxaxises, piAxislist, pfDisancelist):
        '''
        :Description:Absolute multi-axis linear interpolation, supported by controller versions after 20130901.

        :param imaxaxises: Number of participating axes.

        :param piAxislist: Axis list.

        :param pfDisancelist: Distance list.

        :Return: Error code.

        '''
        Axislistarray = (ctypes.c_int * len(piAxislist))(*piAxislist)
        pfDisancelisttarray = (
            ctypes.c_int * len(pfDisancelist))(*pfDisancelist)
        ret = zauxdll.ZAux_Direct_Move(
            self.handle, imaxaxises, Axislistarray, pfDisancelisttarray)
        return ret
