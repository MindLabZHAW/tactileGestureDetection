import time
from frankapy import FrankaArm

def main():
    # 创建FrankaArm实例
    fa = FrankaArm()

    try:
        # 不断获取机器人状态
        while True:
            # 获取当前机器人状态
            robot_state = fa.get_robot_state()

            # 打印一些关节信息作为示例
            print("关节位置: ", robot_state.q)
            print("关节速度: ", robot_state.dq)
            print("末端执行器位置: ", robot_state.O_T_EE)

            # 设置一个间隔时间，例如0.1秒
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("停止机器人状态获取")
    finally:
        # 关闭机器人连接
        fa.stop_skill()

if __name__ == '__main__':
    main()

