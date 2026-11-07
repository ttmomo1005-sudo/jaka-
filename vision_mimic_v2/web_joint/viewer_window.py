"""Windows visibility/focus management for this application's own MuJoCo window.

No other application's window content is inspected or controlled.
"""
import ctypes
from ctypes import wintypes
import os

_pinned = True

def set_pinned(enabled):
    global _pinned
    _pinned = bool(enabled)

def _api():
    if os.name!='nt':return None
    user=ctypes.WinDLL('user32',use_last_error=True)
    user.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowThreadProcessId.restype=wintypes.DWORD
    user.IsWindowVisible.argtypes=[wintypes.HWND];user.IsWindowVisible.restype=wintypes.BOOL
    user.IsIconic.argtypes=[wintypes.HWND];user.IsIconic.restype=wintypes.BOOL
    user.GetWindowTextLengthW.argtypes=[wintypes.HWND];user.GetWindowTextLengthW.restype=ctypes.c_int
    user.GetWindowTextW.argtypes=[wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
    user.GetClassNameW.argtypes=[wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
    user.GetWindowRect.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
    user.MonitorFromWindow.argtypes=[wintypes.HWND,wintypes.DWORD];user.MonitorFromWindow.restype=wintypes.HANDLE
    user.GetForegroundWindow.restype=wintypes.HWND
    user.GetWindowLongPtrW.argtypes=[wintypes.HWND,ctypes.c_int];user.GetWindowLongPtrW.restype=ctypes.c_ssize_t
    user.MonitorFromPoint.argtypes=[wintypes.POINT,wintypes.DWORD];user.MonitorFromPoint.restype=wintypes.HANDLE
    user.ShowWindow.argtypes=[wintypes.HWND,ctypes.c_int];user.ShowWindow.restype=wintypes.BOOL
    user.SetForegroundWindow.argtypes=[wintypes.HWND];user.SetForegroundWindow.restype=wintypes.BOOL
    user.SetWindowPos.argtypes=[wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.UINT]
    user.GetMonitorInfoW.argtypes=[wintypes.HANDLE,ctypes.c_void_p];user.GetMonitorInfoW.restype=wintypes.BOOL
    return user

def windows():
    user=_api()
    if user is None:return []
    result=[];pid=os.getpid()
    callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    @callback_type
    def visit(hwnd,_):
        owner=wintypes.DWORD();user.GetWindowThreadProcessId(hwnd,ctypes.byref(owner))
        if owner.value!=pid:return True
        title=ctypes.create_unicode_buffer(user.GetWindowTextLengthW(hwnd)+1)
        user.GetWindowTextW(hwnd,title,len(title))
        cls=ctypes.create_unicode_buffer(256);user.GetClassNameW(hwnd,cls,256)
        # GLFW also owns an invisible helper/message window. It is not the viewer
        # and must never be restored or used as proof that simulation is visible.
        if 'mujoco' not in title.value.lower() or cls.value!='GLFW30':return True
        rect=wintypes.RECT();user.GetWindowRect(hwnd,ctypes.byref(rect))
        result.append({'handle':int(hwnd),'title':title.value,'class':cls.value,
            'visible':bool(user.IsWindowVisible(hwnd)),'minimized':bool(user.IsIconic(hwnd)),
            'on_screen':bool(user.MonitorFromWindow(hwnd,0)),
            'on_primary':user.MonitorFromWindow(hwnd,0)==user.MonitorFromPoint(wintypes.POINT(0,0),1),
            'topmost':bool(user.GetWindowLongPtrW(hwnd,-20)&0x00000008),
            'foreground':user.GetForegroundWindow()==hwnd,
            'rect':[rect.left,rect.top,rect.right,rect.bottom]})
        return True
    user.EnumWindows.argtypes=[callback_type,wintypes.LPARAM];user.EnumWindows.restype=wintypes.BOOL
    user.EnumWindows(visit,0)
    return result

def status():
    found=windows()
    diagnostics={}
    if os.name=='nt':
        user=_api()
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentThreadId.restype=wintypes.DWORD
        user.GetThreadDesktop.argtypes=[wintypes.DWORD];user.GetThreadDesktop.restype=wintypes.HANDLE
        user.GetProcessWindowStation.restype=wintypes.HANDLE
        user.GetUserObjectInformationW.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
        def name(handle):
            buf=ctypes.create_unicode_buffer(256);needed=wintypes.DWORD()
            return buf.value if user.GetUserObjectInformationW(handle,2,buf,ctypes.sizeof(buf),ctypes.byref(needed)) else None
        diagnostics={'desktop':name(user.GetThreadDesktop(kernel.GetCurrentThreadId())),
                     'window_station':name(user.GetProcessWindowStation())}
        dwm=ctypes.WinDLL('dwmapi')
        dwm.DwmGetWindowAttribute.argtypes=[wintypes.HWND,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
        for w in found:
            cloaked=wintypes.DWORD()
            hr=dwm.DwmGetWindowAttribute(w['handle'],14,ctypes.byref(cloaked),ctypes.sizeof(cloaked))
            w['cloaked']=int(cloaked.value) if hr==0 else None
    interactive = os.name!='nt' or (diagnostics.get('desktop')=='Default' and diagnostics.get('window_station')=='WinSta0')
    return {'supported':os.name=='nt','process_id':os.getpid(),'windows':found,**diagnostics,
        'interactive_desktop':interactive,
        'visible':any(w['visible'] and not w['minimized'] and w['on_screen'] for w in found),
        'pinned':_pinned,
        'presented':interactive and any(w['visible'] and not w['minimized'] and w['on_primary'] and w['topmost'] and w.get('cloaked')==0 for w in found)}

def show():
    user=_api()
    if user is None:return {'supported':False,'visible':True}
    for window in windows():
        hwnd=window['handle']
        user.ShowWindow(hwnd,9)  # SW_RESTORE: unhide/restore only our own viewer.
        class MonitorInfo(ctypes.Structure):
            _fields_=[('size',wintypes.DWORD),('monitor',wintypes.RECT),('work',wintypes.RECT),('flags',wintypes.DWORD)]
        # Explicitly find the primary screen, not the previous window's monitor.
        monitor=user.MonitorFromPoint(wintypes.POINT(0,0),1)
        info=MonitorInfo();info.size=ctypes.sizeof(info)
        if monitor and user.GetMonitorInfoW(monitor,ctypes.byref(info)):
            rect=wintypes.RECT();user.GetWindowRect(hwnd,ctypes.byref(rect));work=info.work
            width=min(900,int((work.right-work.left)*.45))
            height=min(950,int((work.bottom-work.top)*.85))
            x=work.right-width-20
            y=work.top+20
            user.SetWindowPos(hwnd,-1 if _pinned else -2,x,y,width,height,0x0040)
        else:
            user.SetWindowPos(hwnd,-1 if _pinned else -2,0,0,0,0,0x0001|0x0002|0x0040)
        user.SetForegroundWindow(hwnd)
    return status()
