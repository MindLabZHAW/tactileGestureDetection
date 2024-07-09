"""
By Maryam Rezayati

# How to run?

#### 1st  Step: unlock robot
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

#### 2nd Step: run frankapy

open an terminal

	conda activate frankapyenv
	bash robotAPI/frankapy/bash_scripts/start_control_pc.sh -i localhost


#### 3rd Step: run robot node

open another terminal 

	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source robotAPI/franka-interface/catkin_ws/devel/setup.bash --extend
	source robotAPI/frankapy/catkin_ws/devel/setup.bash --extend
	
	$HOME/miniconda/envs/frankapyenv/bin/python3 frankaRobot/main.py

    $HOME/miniconda/envs/frankapyenv/bin/python3 tactileGestureDetection/test.py


# to chage publish rate of frankastate go to : 
sudo nano /franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""

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

