

# 5. 🛠️ 硬件

## 5.1 🎮 遥操作设备

> 下方项目是遥操作时需要使用的设备。

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">项目</th>
    <th style="text-align: center;">数量</th>
    <th style="text-align: center;">规格</th>
    <th style="text-align: center;">备注</th>
  </tr>
  <tr>
    <td style="text-align: center;"><b>宇树通用人形机器人 G1</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;"><a href="">
    <td style="text-align: center;">需选配开发计算单元版本</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>XR 设备</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">
      <a href="https://www.apple.com.cn/apple-vision-pro/">apple-vision-pro</a><br />
      <a href="https://www.picoxr.com/products/pico4-ultra-enterprise">pico4-ultra-enterprise</a><br />
      <a href="https://www.meta.com/quest/quest-3">quest-3</a><br />
      <a href="https://www.meta.com/quest/quest-3s/">quest-3s</a><br />
    </td>
    <td style="text-align: center;">
      Please Refer Our WiKi [XR_Device]
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>路由器</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">推荐至少支持 WiFi6</td>
    <td style="text-align: center;">常规模式必须，无线模式可选。</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>用户电脑</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">推荐 x86-64 架构</td>
    <td style="text-align: center;">
      仿真模式下请使用 
      <a href="https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html"> NVIDIA 官方推荐</a>
      的硬件资源进行部署使用
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>头部相机</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">
      单目相机（机器人内置 Realsense D435i）<br />
      双目相机（支架外置，详情见5.2节表格）
    </td>
    <td style="text-align: center;">
      用于机器人头部视野，双目相机更有沉浸感。<br />
      使用 image_server 文件驱动
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">USB3.0 数据线</td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">
      双直头Type-C，长度0.2米左右
    </td>
    <td style="text-align: center;">
      用于连接头部双目相机
    </td>
  </tr>
</table>


## 5.2 💽 数据采集设备

> 下方项目是录制[数据集]()时的可选设备。设备参数、链接等信息**仅供参考**。

### 5.2.1 双目相机 60 FPS

- 物料

> 该相机与 5.2.2节相机区别是帧率从 30FPS 提升至 60FPS，且安装尺寸有所变化。

|         项目         | 数量 |                             规格                             |                       备注                       |
| :------------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------: |
|     双目RGB相机      |  1   | [60FPS、125°FOV、60mm基线](https://e.tb.cn/h.S2zMNwiUzC9I2H1) |                用于机器人头部视角                |
|     M4x16mm 螺钉     |  2   |           [仅供参考](https://amzn.asia/d/cfta55x)            |                 用于相机支架紧固                 |
| M2x5mm / 6mm自攻螺钉 |  8   |           [仅供参考](https://amzn.asia/d/1msRa5B)            | 用于紧固（相机-相机支架）和（相机支架-相机盖板） |

- 3D打印件

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%; text-align: center;">
  <colgroup>
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;">
  </colgroup>
  <tr>
    <th>项目</th>
    <th>相机支架</th>
    <th>相机盖板</th>
    <th>USB-Type-C 压块</th>
    <th>下载链接</th>
  </tr>
  <tr>
    <td>
      <br />
      <b>经典版头部（98mm）</b>
    </td>
    <td></td>
    <td></td>
    <td align="center"></td>
    <td>📥 经典版3D打印结构件</td>
  </tr>
  <tr>
    <td>
      <br />
      <b>焕新版头部（88mm）</b>
    </td>
    <td></td>
    <td></td>
    <td align="center"></td>
    <td>📥 焕新版3D打印结构件</td>
  </tr>
</table>

### 5.2.2 双目相机 30 FPS

- 物料

|         项目         | 数量 |                             规格                             |                       备注                       |
| :------------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------: |
|       双目相机       |  1   | [30FPS、125°FOV、60mm基线](http://e.tb.cn/h.TaZxgkpfWkNCakg) |                用于机器人头部视角                |
|     M4x16mm 螺钉     |  2   |           [仅供参考](https://amzn.asia/d/cfta55x)            |                 用于相机支架紧固                 |
| M2x5mm / 6mm自攻螺钉 |  8   |           [仅供参考](https://amzn.asia/d/1msRa5B)            | 用于紧固（相机-相机支架）和（相机支架-相机盖板） |

- 3D打印件

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%; text-align: center;">
  <colgroup>
    <col style="width: 20%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 30%;">
  </colgroup>
  <tr>
    <th>项目</th>
    <th>相机支架</th>
    <th>相机盖板</th>
    <th>下载链接</th>
  </tr>
  <tr>
    <td>
      <br />
      <b>经典版头部（98mm）</b>
    </td>
    <td></td>
    <td>无</td>
    <td>📥 经典版3D打印结构件</td>
  </tr>
  <tr>
    <td>
      <br />
      <b>焕新版头部（88mm）</b>
    </td>
    <td></td>
    <td></td>
    <td>📥 焕新版3D打印结构件</td>
  </tr>
</table>

### 5.2.3 G1 腕部 RealSense D405

>  RealSense D405 仅推荐 [Dex3-1]() 末端执行器使用

- 物料

|        项目         | 数量 |                             规格                             |                             备注                             |
| :-----------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| RealSense D405 相机 |  2   | [Website](https://www.intelrealsense.com/depth-camera-d405/) |          用于G1机器人（M4010腕部电机）左右腕部视角           |
|     USB3.0 Hub      |  1   | [Issue](https://github.com/IntelRealSense/librealsense/issues/24) | 注意选用优质Hub以满足realsense相机需求；<br />推荐连接至 [9号Type-C接口]() |
|    M3-1 六角螺母    |  4   |              [仅供参考](https://a.co/d/gQaLtHD)              |                         用于紧固腕部                         |
|     M3x12 螺钉      |  4   |           [仅供参考](https://amzn.asia/d/aU9NHSf)            |                         用于紧固腕部                         |
|      M3x6 螺钉      |  4   |           [仅供参考](https://amzn.asia/d/0nEz5dJ)            |                         用于紧固腕部                         |

- 3D打印件

|       项目        | 数量 |          备注          |                           下载链接                           |
| :---------------: | :--: | :--------------------: | :----------------------------------------------------------: |
| D405 相机腕圈支架 |  2   | 与腕部相机支架搭配使用 | [📥 STEP]() |
|   左腕相机支架    |  1   |  用于装配左腕D405相机  | [📥 STEP]() |
|   右腕相机支架    |  1   |  用于装配右腕D405相机  | [📥 STEP]() |

### 5.2.4 G1 腕部单目相机

- 物料

|         项目         | 数量 |                             规格                             |                       备注                       |
| :------------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------: |
|       单目相机       |  2   | [60FPS、160° FOV](https://e.tb.cn/h.S2YWUJan6ZP8Wqv?tk=MqHK4uvWlLk) |    用于G1机器人（M4010腕部电机）左右腕部视角     |
|      USB3.0 Hub      |  1   | [仅供参考](https://e.tb.cn/h.S2QB8hVuKbNfqb9?tk=XsBL4uwn2Ch) |               用于连接两路腕部相机               |
|    M3-1 六角螺母     |  4   |              [仅供参考](https://a.co/d/gQaLtHD)              |                  用于腕部紧固件                  |
|      M3x12 螺钉      |  4   |           [仅供参考](https://amzn.asia/d/aU9NHSf)            |              用于紧固腕部支架和腕圈              |
|   M2.5x5 自攻螺钉    |  4   |           [仅供参考](https://amzn.asia/d/0nEz5dJ)            |              用于紧固线卡和腕部支架              |
| M2x5mm / 6mm自攻螺钉 |  8   |           [仅供参考](https://amzn.asia/d/1msRa5B)            | 用于紧固（相机-相机支架）和（相机支架-相机盖板） |

- 3D打印件

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr>
    <th>末端执行器</th>
    <th>相机支架</th>
    <th>腕圈支架</th>
    <th>相机盖板</th>
    <th>线卡</th>
    <th>下载链接</th>
  </tr>
  <tr>
    <td>Dex1-1</td>
    <td></td>
    <td></td>
    <td></td>
    <td rowspan="3" valign="middle">
      
    </td>
    <td rowspan="3" valign="middle">
      📥 3D打印件下载链接
    </td>
  </tr>
  <tr>
    <td>Dex3-1</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>
      Inspire DFX Hand /
      Brainco Hand
    </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

## 5.3 🔨 安装示意图（部分）

<table>
    <tr>
        <th align="center">项目</th>
        <th align="center" colspan="2">仿真</th>
        <th align="center" colspan="2">实物</th>
    </tr>
    <tr>
        <td align="center">头部</td>
        <td align="center">
            <p align="center">
                <img src="./img/head_camera_mount.png" alt="head" width="100%">
                <figcaption>头部支架</figcaption>
            </p>
        </td>
        <td align="center">
            <p align="center">
                <img src="./img/head_camera_mount_install.png" alt="head" width="80%">
                <figcaption>装配侧视</figcaption>
            </p>
        </td>
        <td align="center" colspan="2">
            <p align="center">
                <img src="./img/real_head.jpg" alt="head" width="20%">
                <figcaption>装配正视</figcaption>
            </p>
        </td>
    </tr>
    <tr>
        <td align="center">腕部</td>
        <td align="center" colspan="2">
            <p align="center">
                <img src="./img/wrist_and_ring_mount.png" alt="wrist" width="100%">
                <figcaption>腕圈及相机支架</figcaption>
            </p>
        </td>
        <td align="center">
            <p align="center">
                <img src="./img/real_left_hand.jpg" alt="wrist" width="50%">
                <figcaption>装配右手</figcaption>
            </p>
        </td>
        <td align="center">
            <p align="center">
                <img src="./img/real_right_hand.jpg" alt="wrist" width="50%">
                <figcaption>装配左手</figcaption>
            </p>
        </td>
    </tr>
</table>

> 注意：如图中红圈所示，腕圈支架与机器人手腕接缝对齐。
