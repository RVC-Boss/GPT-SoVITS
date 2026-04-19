import time
#time styles
STYLE_Y = "%Y"
STYLE_M = "%m"
STYLE_D = "%d"
STYLE_H = "%H"
STYLE_MIN = "%M"
STYLE_S = "%S"
STYLE_FULL = "%Y-%m-%d_%H.%M.%S"
#quick calls
def get_time_y(STYLE = STYLE_Y):
    return time.strftime(STYLE, time.localtime())
def get_time_m(STYLE = STYLE_M):
    return time.strftime(STYLE, time.localtime())
def get_time_d(STYLE = STYLE_D):
    return time.strftime(STYLE, time.localtime())
def get_time_h(STYLE = STYLE_H):
    return time.strftime(STYLE, time.localtime())
def get_time_min(STYLE = STYLE_MIN):
    return time.strftime(STYLE, time.localtime())
def get_time_s(STYLE = STYLE_S):
    return time.strftime(STYLE, time.localtime())
def get_time_full(STYLE = STYLE_FULL):
    return time.strftime(STYLE, time.localtime())

def s(t:float):
    time.sleep(t)
    return 
###

if __name__ == '__main__':
    print(get_time_y())
    print(get_time_m())
    print(get_time_d())
    print(get_time_h())
    print(get_time_min())
    print(get_time_s())
    print(get_time_full())