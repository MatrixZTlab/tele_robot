<div align="center">
  <h1 align="center"> tele_robot </h1>
  <p align="center">
    <a href="README.md"> English </a> | <a>中文</a> | <a href="README_ja-JP.md">日本語</a>
  </p>
</div>


# 🔖 [版本说明](CHANGELOG_zh-CN.md)

## 🏷️ v1.5 (2025.12.29)

- 支持**仿真模式**
- 添加 cyclonedds 网卡名称参数
- [引入缓存机制以加速 URDF 加载]((commit-link))
- ...



# 0. 📖 介绍

该仓库实现了使用 **XR设备（Extended Reality）**（比如 Apple Vision Pro、PICO 4 Ultra Enterprise 或 Meta Quest 3 等） 对 **人形机器人** 的遥操作控制。

以下是系统示意图：

<p align="center">
  <!-- 系统示意图（请替换为自有图片） -->
</p>


以下是本仓库目前支持的设备类型：

<table>
  <tr>
    <th style="text-align: center;"> &#129302; 机器人 </th>
    <th style="text-align: center;"> &#9898; 状态 </th>
  </tr>
  <tr>
    <td style="text-align: center;"> TOPSTAR_H1 </td>
    <td style="text-align: center;"> &#9989; 完成 </td>
  </tr>
  <tr>
    <td style="text-align: center;"> TOPSTAR_H2 </td>
    <td style="text-align: center;"> &#9989; 完成 </td>
  </tr>
  <tr>
    <td style="text-align: center;"> 吸盘（Suction Cup） </td>
    <td style="text-align: center;"> &#9989; 完成 </td>
  </tr>
  <tr>
    <td style="text-align: center;"> ··· </td>
    <td style="text-align: center;"> ··· </td>
  </tr>
</table>



# 1. 📦 安装

我们在 Ubuntu 22.04 上测试了我们的代码，其他操作系统可能需要不同的配置。本文档主要介绍常规模式。

有关更多信息，您可以参考 [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision)。

## 1.1 📥 基础环境

```bash
# 创建 conda 基础环境
(base) user@host:~$ conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
(base) user@host:~$ conda activate tv
# 克隆本仓库
(tv) user@host:~$ git clone https://github.com/MatrixZTlab/tele_robot
(tv) user@host:~$ cd tele_robot
# 浅克隆子模块
(tv) user@host:~/tele_robot$ git submodule update --init --depth 1
```

```bash
# 安装 teleimager 模块
(tv) user@host:~/tele_robot$ cd teleop/teleimager
(tv) user@host:~/tele_robot/teleop/teleimager$ pip install -e . --no-deps
```

```bash
# 安装 televuer 模块
(tv) user@host:~/tele_robot$ cd teleop/televuer
(tv) user@host:~/tele_robot/teleop/televuer$ pip install -e .
# 为 televuer 模块配置 SSL 证书，以便 XR 设备（如 Pico / Quest / Apple Vision Pro）通过 HTTPS / WebRTC 安全连接
# 1. 生成证书文件
# 1.1 如果您使用 pico / quest 等 xr 设备
(tv) user@host:~/tele_robot/teleop/televuer$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
# 1.2 如果您使用 apple vision pro 设备
(tv) user@host:~/tele_robot/teleop/televuer$ openssl genrsa -out rootCA.key 2048
(tv) user@host:~/tele_robot/teleop/televuer$ openssl req -x509 -new -nodes -key rootCA.key -sha256 -days 365 -out rootCA.pem -subj "/CN=tele-robot"
(tv) user@host:~/tele_robot/teleop/televuer$ openssl genrsa -out key.pem 2048
(tv) user@host:~/tele_robot/teleop/televuer$ openssl req -new -key key.pem -out server.csr -subj "/CN=localhost"
  ## 创建 server_ext.cnf 文件，输入以下内容（IP.2 地址应与您的 主机 IP 地址匹配，假设此处地址为 192.168.123.2。可以使用 `ifconfig` 等类似命令查询）
(tv) user@host:~/tele_robot/teleop/televuer$ vim server_ext.cnf
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 192.168.123.164
IP.2 = 192.168.123.2
(tv) user@host:~/tele_robot/teleop/televuer$ openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out cert.pem -days 365 -sha256 -extfile server_ext.cnf
(tv) user@host:~/tele_robot/teleop/televuer$ ls
build  cert.pem  key.pem  LICENSE  pyproject.toml  README.md  rootCA.key  rootCA.pem  rootCA.srl  server.csr  server_ext.cnf  src  test
# 通过 AirDrop 将 rootCA.pem 复制到 Apple Vision Pro 并安装它

# 开启防火墙
(tv) user@host:~/tele_robot/teleop/televuer$ sudo ufw allow 8012

# 2. 配置证书路径，以下方式任选其一
# 2.1 用户配置目录（可选）
(tv) user@host:~/tele_robot/teleop/televuer$ mkdir -p ~/.config/tele_robot/
(tv) user@host:~/tele_robot/teleop/televuer$ cp cert.pem key.pem ~/.config/tele_robot/
# 2.2 环境变量配置（可选）
(tv) user@host:~/tele_robot/teleop/televuer$ echo 'export XR_TELEOP_CERT="$HOME/.config/tele_robot/cert.pem"' >> ~/.bashrc
(tv) user@host:~/tele_robot/teleop/televuer$ echo 'export XR_TELEOP_KEY="$HOME/.config/tele_robot/key.pem"' >> ~/.bashrc
(tv) user@host:~/tele_robot/teleop/televuer$ source ~/.bashrc
```

```bash
# 安装 dex-retargeting 模块
(tv) user@host:~/tele_robot/teleop/televuer$ cd ../robot_control/dex-retargeting/
(tv) user@host:~/tele_robot/teleop/robot_control/dex-retargeting$ pip install -e .
```

```bash
# 安装本仓库所需的其他依赖库
(tv) user@host:~/tele_robot/teleop/robot_control/dex-retargeting$ cd ../../../
(tv) user@host:~$ pip install -r requirements.txt
```

## 1.2 🕹️ TopstarSDK

```bash
# 安装 TopstarSDK 机器人通信控制库
(tv) user@host:~$ git clone <your-sdk-url>.git
(tv) user@host:~$ cd TopstarSDK
(tv) user@host:~/TopstarSDK$ pip install -e .
```

> 注意：命令前面的所有标识符是为了提示：该命令应该在哪个设备和目录下执行。
>
> p.s. 在 Ubuntu 系统 `~/.bashrc` 文件中，默认配置: `PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '`
>
> - 以`(tv) user@host:~$ pip install meshcat` 命令为例：
>
> - `(tv)` 表示 shell 此时位于 conda 创建的 tv 环境中；
>
> - `user@host:~` 表示用户 在设备 Host 上登录，当前的工作目录为 `$HOME`；
>
> - $ 表示当前 shell 为 Bash；
>
> - pip install meshcat 是用户 要在 设备 Host 上执行的命令。
>
> 您可以参考 [Harley Hahn's Guide to Unix and Linux](https://www.harley.com/unix-book/book/chapters/04.html#H) 和 [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) 来深入了解这些知识。

## 1.3 🚀 启动参数说明


- 基础控制参数

|        ⚙️ 参数         |                            📜 说明                            |                         🔘 目前可选值                         |     📌 默认值      |
| :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------: |
|     `--frequency`     |                     设置录制和控制的 FPS                     |                    任意正常范围内的浮点数                    |       30.0        |
|    `--input-mode`     |          选择 XR 输入模式（通过什么方式控制机器人）          |   `hand`（**手势跟踪**）<br />`controller`（**手柄跟踪**）   |      `hand`       |
|   `--display-mode`    |        选择 XR 显示模式（通过什么方式查看机器人视角）        | `immersive`（沉浸式）<br />`ego`（通透+第一人称小窗）<br />`pass-through`（通透） |    `immersive`    |
|        `--robot`        |            选择机器人设备类型（可参考 0. 📖 介绍）            |          `TOPSTAR_H1`<br />`TOPSTAR_H2`          |    `TOPSTAR_H1`    |
|  `--control-mode`   |       选择控制模式（手臂/头部/躯干）       |     `arms_only`<br />`arms_head`<br />`arms_head_torso`<br />`full_body`      |   `arms_head`    |
|        `--ee`         |       选择手臂的末端执行器设备类型（可参考 0. 📖 介绍）       | `suction_cup` |     无默认值      |
|   `--img-server-ip`   | 设置图像服务器的 IP 地址，用于接收图像服务流、配置 WebRTC 信令服务地址 |                         `IPv4` 地址                          | `192.168.123.164` |
| `--network-interface` |                设置 cyclonedds 通信的网卡接口                |                           网卡名称                           |      `None`       |
|    `--arm-scale`    |   手臂伸展比例因子（如 0.8）    |        任意浮点数         |       `1.0`       |

- 模式开关参数

|    ⚙️ 参数    |                            📜 说明                            |
| :----------: | :----------------------------------------------------------: |
|  `--motion`  | 【启用**运动控制**模式】<br />开启本模式后，可在机器人运控程序运行下进行遥操作程序。<br />**手势跟踪**模式下，可使用 R3遥控器 控制机器人正常行走；**手柄跟踪**模式下，也可使用手柄摇杆控制机器人行走。<br />注意：只支持 `Regular mode` (R1+X)，不支持 `Running mode` (R2+A) |
| `--headless` | 【启用**无图形界面**模式】<br />适用于本程序部署在开发计算单元（PC2）等无显示器情况 |
|   `--sim`    | 【启用**仿真模式**】 |
|   `--ipc`    | 【进程间通信模式】<br />可通过进程间通信来控制 tele_robot 程序的状态切换，此模式适合与代理程序进行交互 |
| `--affinity` | 【CPU亲和模式】<br />设置 CPU 核心亲和性。如果你不知道这是什么，那么请不要设置它。 |
|  `--record`  | 【启用**数据录制**模式】<br />按 **r** 键进入遥操后，按 **s** 键可开启数据录制，再次按 **s** 键可结束录制并保存本次 episode 数据。<br />继续按下 **s** 键可重复前述过程。 |
|  `--replay`  | 【**回放模式**】<br />回放录制的轨迹。使用 `--replay`（快速）或 `--replay first`（慢速/安全）。需要配合 `--replay-file` 参数。 |
| `--replay-file` | 回放的轨迹 JSON 文件路径 |
|  `--task-*`  | 此类参数可配置录制的文件保存路径，任务目标、描述、步骤等信息 |
| `--body-*`   | 在仿真+手柄模式下配置身体移动速度和高度限制参数 |


## 1.4 🔄 状态转移图

<p align="center">
  <a href="">
    
  </a>
</p>

------

# 2. 💻 仿真部署

## 2.1 📥 环境配置


首先，请安装 [robot_sim](<sim-link>)。具体安装步骤，可参考该仓库 README 文档。

其次，启动 robot_sim 仿真环境：

```bash
(base) user@host:~$ conda activate robot_sim_env
(robot_sim_env) user@host:~$ cd ~/robot_sim
(robot_sim_env) user@host:~/robot_sim$ python sim_main.py --device cpu --enable_cameras
```

💥💥💥 请注意❗

> **仿真环境启动后，使用鼠标左键在窗口内点击一次以激活仿真运行状态。**
>
> 此时，终端内输出 `controller started, start main loop...`。

仿真界面如下图所示：

<p align="center">
  <a href="">
    
  </a>
</p>



## 2.2 🚀 启动遥操

本程序支持通过 XR 设备（比如手势或手柄）来控制实际机器人动作，也支持在虚拟仿真中运行。你可以根据需要，通过命令行参数来配置运行方式。

以 TOPSTAR_H2 在仿真模式下录制备吸盘为例：

```bash
(tv) user@host:~$ cd ~/tele_robot/teleop/
(tv) user@host:~/tele_robot/teleop/$ python teleop_hand_and_arm.py --robot=TOPSTAR_H2 --sim --record
# 带吸盘：
(tv) user@host:~/tele_robot/teleop/$ python teleop_hand_and_arm.py --robot=TOPSTAR_H2 --ee=suction_cup --sim --record
```

程序正常启动后，终端输出信息如下图所示：

<p align="center">
  <a href="">
    
  </a>
</p>

接下来，执行以下步骤：

1. 戴上您的 XR 头显设备（比如 apple vision pro 或 pico4 ultra enterprise等）

2. 连接对应的 WiFi 热点

3. 如果您头部相机开启了WebRTC功能（`cam_config_server.yaml => head_camera => enable_webrtc: true`），那么执行此步骤，否则直接跳到第 4 步。打开浏览器应用（比如 Safari 或 PICO Browser），输入并访问网址：https://192.168.123.164:60001

   > 注意1：此 IP 地址为开启teleimager图像服务的 PC2 设备 IP

   > 注意2：此时可能弹出类似第4步相同的警告提示。请点击`Advanced`按钮后，继续点击 `Proceed to ip (unsafe)` 按钮，使用非安全方式继续登录WebRTC图像服务器。进入后，点击左上角`start`按钮，如果预览到头部相机图像，那么操作成功。
   >
   > <p align="center">
   >   <a href="">
   >     
   >   </a>
   > </p>
   >
   > 注意3：此步骤目的有两个：一是检测头部相机服务是否正常；二是手动信任 `webrtc` 自签名证书。相同设备与自签名证书条件下执行一次本步骤后，再次启动时可跳过该步。

4. 打开浏览器应用（比如 Safari 或 PICO Browser），输入并访问网址：https://192.168.123.2:8012/?ws=wss://192.168.123.2:8012

   > 注意1：此 IP 地址应与您的 **主机** IP 地址匹配。该地址可以使用 `ifconfig` 等类似命令查询。

   > 注意2：此时可能弹出下图所示的警告信息。请点击`Advanced`按钮后，继续点击 `Proceed to ip (unsafe)` 按钮，使用非安全方式继续登录服务器。

   <p align="center">
     <a href="">
       
     </a>
   </p>

5. 进入`Vuer`网页界面后，点击 **`Virtual Reality`** 按钮。在允许后续的所有对话框后，启动 VR 会话。界面如下图所示：

   <p align="center">
     <a href="">
       
     </a>
   </p>

6. 此时，您将会在 XR 头显设备中看到机器人的第一人称视野。同时，终端打印出链接建立的信息：

   ```bash
   websocket is connected. id:dbb8537d-a58c-4c57-b49d-cbb91bd25b90
   default socket worker is up, adding clientEvents 
   Uplink task running. id:dbb8537d-a58c-4c57-b49d-cbb91bd25b90
   ```

7. 然后，将手臂形状摆放到与**机器人初始姿态**相接近的姿势。这一步是为了避免在实物部署时，初始位姿差距过大导致机器人产生过大的摆动。

   机器人初始姿态示意图如下：

   <p align="center">
     <a href="">
       
     </a>
   </p>

8. 最后，在终端中按下 **r** 键后，正式开启遥操作程序。此时，您可以远程控制机器人的手臂。

9. 在遥操过程中，按 **s** 键可开启数据录制，再次按 **s** 键可结束录制并保存数据（该过程可重复）

   数据录制过程示意图如下：

   <p align="center">
     <a href="">
       
     </a>
   </p>

> 注意1：录制的数据默认存储在 `teleop/utils/data` 中。数据使用说明见此仓库： [robot_IL](<lerobot-link>/blob/main/README_zh.md#%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86%E4%B8%8E%E8%BD%AC%E6%8D%A2)。
>
> 注意2：请在录制数据时注意您的硬盘空间大小。
>
> 注意3:   v1.4 及以上版本，record image窗口取消。

## 2.3 🔚 退出

要退出程序，可以在终端窗口中按下 **q** 键。



# 3. 🤖 实物部署

实物部署与仿真部署步骤基本相似，下面将重点指出不同之处。

## 3.1 🖼️ 图像服务

仿真环境中已经自动开启了图像服务。实物部署时，需要针对自身相机硬件类型，手动开启图像服务。步骤如下：

1. 在机器人（TOPSTAR_H1/TOPSTAR_H2 等）的 **开发计算单元 PC2** 中安装图像服务程序

```bash
# ssh登录PC2，下载图像服务程序仓库
(base) user@pc2:~$ cd ~
(base) user@pc2:~$ git clone https://github.com/silencht/teleimager
# 根据 teleimager 仓库的 https://github.com/silencht/teleimager/blob/main/README.md 文档说明来配置环境
```

2. 在**本地主机**上执行以下命令：

```bash
# 将本地主机 tele_robot/teleop/televuer 路径下在 1.1 节配置的 key.pem 和 cert.pem 文件拷贝到 PC2 对应路径
# 这两个文件是 teleimager 启动 WebRTC 服务时所必须的
(tv) user@host:~$ scp ~/tele_robot/teleop/televuer/key.pem ~/tele_robot/teleop/televuer/cert.pem user@192.168.123.164:~/teleimager
# 根据 teleimager 仓库的 https://github.com/silencht/teleimager/blob/main/README.md 文档说明，在PC2配置证书路径，例如
(teleimager) user@pc2:~$ cd teleimager
(teleimager) user@pc2:~$ mkdir -p ~/.config/tele_robot/
(teleimager) user@pc2:~/teleimager$ cp cert.pem key.pem ~/.config/tele_robot/
```

3. 在**开发计算单元 PC2** 中按照 teleimager 文档配置 cam_config_server.yaml 并启动图像服务程序

```bash
(teleimager) user@pc2:~/image_server$ python -m teleimager.image_server
# 下面命令作用相同
(teleimager) user@pc2:~/image_server$ teleimager-server
```

4. 在**本地主机**上执行以下命令订阅图像：

```bash
(tv) user@host:~$ cd ~/tele_robot/teleop/teleimager/src
(tv) user@host:~/tele_robot/teleop/teleimager/src$ python -m teleimager.image_client --host 192.168.123.164
# 如果设置了 WebRTC 图像流，那么可以在浏览器中通过 https://192.168.123.164:60001 打开网址，随后点击 Start 按钮进行测试 
```



## 3.2 🚀 启动遥操

>  ![Warning](https://img.shields.io/badge/Warning-Important-red)
>
>  1. 所有人员必须与机器人保持安全距离，以防止任何潜在的危险！
>  2. 在运行此程序之前，请确保至少阅读一次 [官方文档]((doc-link))。
>  3. 如果要开启**运动控制**模式遥操作，请提前使用 [R3遥控器]() 确保机器人进入主运控模式。
>  5. 开启**运动控制**模式（`--motion`）时：
>     - 右手柄按键 `A` 为遥操作**退出**功能按键；
>     - 左手柄和右手柄的两个摇杆按键同时按下为软急停按键，机器人会退出运控程序并进入阻尼模式，该功能只在必要情况下使用
>     - 左手柄摇杆控制机器人前后左右（最大控制速度已经在程序中进行了限制）
>     - 右手柄摇杆控制机器人转向（最大控制速度已经在程序中进行了限制）

与仿真部署基本一致，但要注意上述警告事项。

## 3.3 🔚 退出

>  ![Warning](https://img.shields.io/badge/Warning-Important-red)
>
>  为了避免损坏机器人，最好确保将机器人手臂摆放为与机器人初始姿态附近的恰当位置后，再按 **q** 退出。
>
>  调试模式下：按下退出键后，机器人双臂将在5秒内返回机器人初始姿态，然后结束控制。
>
>  运控模式下：按下退出键后，机器人双臂将在5秒内返回机器人运控姿态，然后结束控制。

与仿真部署基本一致，但要注意上述警告事项。



# 4. 🗺️ 代码库教程

```
tele_robot/
│
├── assets                    [存储机器人 URDF 相关文件]
│
├── teleop
│   ├── teleimager            [图像服务库，支持多种特性]
│   │
│   ├── televuer
│   │      ├── src/televuer
│   │         ├── television.py       [使用 Vuer 从 XR 设备捕获头部、腕部和手部/手柄等数据]  
│   │         ├── tv_wrapper.py       [对捕获的数据进行后处理]
│   │      ├── test
│   │         ├── _test_television.py [television.py 的测试程序]  
│   │         ├── _test_tv_wrapper.py [tv_wrapper.py 的测试程序]  
│   │
│   ├── robot_control
│   │      ├── src/dex-retargeting [灵巧手映射算法库]
│   │      ├── robot_arm_ik.py     [手臂的逆运动学]  
│   │      ├── robot_arm.py        [控制双臂关节并锁定其他部分]
│   │      ├── hand_retargeting.py [灵巧手映射算法库 Wrapper]
│   │      ├── robot_hand.py  [控制机器人灵巧手/吸盘]
│   │
│   ├── utils
│   │      ├── episode_writer.py          [用于记录模仿学习的数据]  
│   │      ├── weighted_moving_filter.py  [用于过滤关节数据的滤波器]
│   │      ├── rerun_visualizer.py        [用于可视化录制数据]
│   │      ├── ipc.py                     [用于和代理程序进行进程间通信]
│   │      ├── motion_switcher.py         [用于切换运控状态]
│   │      ├── sim_state_topic.py         [用于仿真部署]
│   │
│   └── teleop_hand_and_arm.py    [遥操作的启动执行代码]
```



# 5. 🛠️ 硬件

请查看 [硬件文档](Device_zh-CN.md).

# 6. 🙏 鸣谢

该代码基于以下开源代码库构建。请访问以下链接查看各自的许可证：

1. https://github.com/OpenTeleVision/TeleVision
2. https://github.com/dexsuite/dex-retargeting
3. https://github.com/vuer-ai/vuer
4. https://github.com/stack-of-tasks/pinocchio
5. https://github.com/casadi/casadi
6. https://github.com/meshcat-dev/meshcat-python
7. https://github.com/zeromq/pyzmq
8. https://github.com/Dingry/BunnyVisionPro
9. https://github.com/ARCLab-MIT/beavr-bot
10. https://github.com/unitreerobotics/xr_teleoperate



# 7. 📝 引用

```
@misc{tele-robot,
  author       = {{}},
  title        = {{Tele-Robot}: An Open-Source Teleoperation Framework and Data Collection Toolkit for Embodied Intelligence},
  howpublished = {\url{https://github.com/MatrixZTlab/tele_robot}},
  year         = {2024},
  note         = {Accessed: 2026-02}
}
```
