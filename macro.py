from MAMEToolkit.sf_environment.Actions import Actions
    
#### 连续技
# 膝蹴 
def comb0():
    steps = [
        {"hold": 3, "actions": [Actions.P1_JPUNCH, Actions.P1_SKICK]}]
    return steps

# 背负投
def comb1():
    steps = [
        {"hold": 3, "actions": [Actions.P1_RIGHT, Actions.P1_JPUNCH, Actions.P1_SKICK]}]
    return steps

#  地狱车
def comb2():
    steps = [
        {"hold": 3, "actions": [Actions.P1_LEFT, Actions.P1_JPUNCH, Actions.P1_SKICK]}]
    return steps

# TC
def comb3():
    steps = [
        {"hold": 3, "actions": [Actions.P1_SPUNCH, Actions.P1_FPUNCH]}]
    return steps

# 踏进前蹴  
def comb4():
    steps = [
        {"hold": 3, "actions": [Actions.P1_RIGHT, Actions.P1_FKICK]}]
    return steps

def comb4_():
    steps = [
        {"hold": 3, "actions": [Actions.P1_LEFT, Actions.P1_FKICK]}]
    return steps

# 紫电踵落 
def comb5():
    steps = [
        {"hold": 3, "actions": [Actions.P1_RIGHT, Actions.P1_RKICK]}]
    return steps

def comb5_():
    steps = [
        {"hold": 3, "actions": [Actions.P1_LEFT, Actions.P1_RKICK]}]
    return steps

# 波动拳(EX/SC)
def comb6():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT]},
        {"hold": 1, "actions": [Actions.P1_SPUNCH]}]
    return steps

def comb6_():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT]},
        {"hold": 1, "actions": [Actions.P1_SPUNCH]}]
    return steps

# 升龙拳(EX/SC)  
def comb7():
    steps = [
        {"hold": 1, "actions": [Actions.P1_RIGHT]},
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_JPUNCH]}]
    return steps

def comb7_():
    steps = [
        {"hold": 1, "actions": [Actions.P1_LEFT]},
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_JPUNCH]}]
    return steps

# 龙卷旋风脚(EX) 
def comb8():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT]},
        {"hold": 1, "actions": [Actions.P1_SKICK]}]
    return steps

def comb8_():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT]},
        {"hold": 1, "actions": [Actions.P1_SKICK]}]
    return steps

# 升龙裂破(EX)  
def comb9():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT]},
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT]},
        {"hold": 1, "actions": [Actions.P1_JPUNCH]}]
    return steps

def comb9_():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT]},
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT]},
        {"hold": 1, "actions": [Actions.P1_JPUNCH]}]
    return steps

# 神龙拳 (EX)  
def comb10():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT]},
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_RIGHT]},
        {"hold": 3, "actions": [Actions.P1_SKICK]}]
    return steps

def comb10_():
    steps = [
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT]},
        {"hold": 1, "actions": [Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]},
        {"hold": 1, "actions": [Actions.P1_LEFT]},
        {"hold": 3, "actions": [Actions.P1_SKICK]}]
    return steps

#### 小动作
# 跳+攻击
def comb11():
    steps = [
        {"hold": 3, "actions": [Actions.P1_UP, Actions.P1_SKICK]}]
    return steps

def comb11_():
    steps = [
        {"hold": 3, "actions": [Actions.P1_UP, Actions.P1_SKICK]}]
    return steps

def comb12():
    steps = [
        {"hold": 3, "actions": [Actions.P1_UP, Actions.P1_JPUNCH]}]
    return steps

def comb12_():
    steps = [
        {"hold": 3, "actions": [Actions.P1_UP, Actions.P1_JPUNCH]}]
    return steps

# 蹲下攻击
def comb13():
    steps = [
        {"hold": 3, "actions": [Actions.P1_DOWN, Actions.P1_JPUNCH]}]
    return steps

def comb13_():
    steps = [
        {"hold": 3, "actions": [Actions.P1_DOWN, Actions.P1_FPUNCH]}]
    return steps

def comb14():
    steps = [
        {"hold": 3, "actions": [Actions.P1_DOWN, Actions.P1_SKICK]}]
    return steps

def comb14_():
    steps = [
        {"hold": 3, "actions": [Actions.P1_DOWN, Actions.P1_RKICK]}]
    return steps

# 防御
def comb15():
    steps = [
        {"hold": 8, "actions": [Actions.P1_LEFT]}]
    return steps

def comb15_():
    steps = [
        {"hold": 8, "actions": [Actions.P1_RIGHT]}]
    return steps

def comb16():
    steps = [
        {"hold": 8, "actions": [Actions.P1_LEFT, Actions.P1_DOWN]}]
    return steps

def comb16_():
    steps = [
        {"hold": 8, "actions": [Actions.P1_RIGHT, Actions.P1_DOWN]}]
    return steps

# 移动
def comb17():
    steps = [
        {"hold": 8, "actions": [Actions.P1_LEFT]}]
    return steps

def comb17_():
    steps = [
        {"hold": 8, "actions": [Actions.P1_RIGHT]}]
    return steps

def comb18():
    steps = [
        {"hold": 8, "actions": [Actions.P1_LEFT, Actions.P1_UP]}]
    return steps

def comb18_():
    steps = [
        {"hold": 8, "actions": [Actions.P1_RIGHT,  Actions.P1_UP]}]
    return steps


index_to_comb = (
     comb0,
     comb1,
     comb2,
     comb3,
     comb4,
     comb4_,
     comb5,
     comb5_,
     comb6,
     comb6_,
     comb7,
     comb7_,
     comb8,
     comb8_,
     comb9,
     comb9_,
     comb10,
     comb10_,
     comb11,
     comb11_,
     comb12,
     comb12_,
     comb13,
     comb13_,
     comb14,
     comb14_,
     comb15,
     comb15_,
     comb16,
     comb16_,
     comb17,
     comb17_,
     comb18,
     comb18_
)

MACRO_NUMS = len(index_to_comb)