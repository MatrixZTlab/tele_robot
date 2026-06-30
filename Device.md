# 5. 🛠️ Hardware

## 5.1 🎮 Teleoperation Devices

> The following items are required for teleoperation.

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Item</th>
    <th style="text-align: center;">Quantity</th>
    <th style="text-align: center;">Specification</th>
    <th style="text-align: center;">Remarks</th>
  </tr>
  <tr>
    <td style="text-align: center;"><b>TOPSTAR Humanoid Robot</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">TOPSTAR_H1 / TOPSTAR_H2</td>
    <td style="text-align: center;">Developer computing unit version required</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>XR Device</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">
      <a href="https://www.apple.com.cn/apple-vision-pro/">apple-vision-pro</a><br />
      <a href="https://www.picoxr.com/products/pico4-ultra-enterprise">pico4-ultra-enterprise</a><br />
      <a href="https://www.meta.com/quest/quest-3">quest-3</a><br />
      <a href="https://www.meta.com/quest/quest-3s/">quest-3s</a><br />
    </td>
    <td style="text-align: center;">
      Please refer to your device documentation
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Router</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">Recommended: at least WiFi6 support</td>
    <td style="text-align: center;">Required in wired mode; optional in wireless mode.</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>User Computer</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">Recommended x86-64 architecture</td>
    <td style="text-align: center;">
      For simulation mode, please follow 
      <a href="https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html">NVIDIA official hardware recommendations</a>
      for deployment.
    </td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Head Camera</b></td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">
      Monocular camera (built-in Realsense D435i)<br />
      Stereo camera (external mount, see details at chapter 5.2)
    </td>
    <td style="text-align: center;">
      Used for robot head perspective, stereo camera provides more immersion.<br />
      Driven by image_server module
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">USB3.0 Cable</td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">
      Type-C double straight connectors, about 0.2m length
    </td>
    <td style="text-align: center;">
      For connecting the stereo head camera
    </td>
  </tr>
</table>


## 5.2 💽 Data Collection Devices

> The following items are optional devices for recording datasets. Parameters, links, etc. are for **reference only**.

### 5.2.1 Stereo Camera 60 FPS

- Materials

> Compared with the camera in Section 5.2.2, this one increases the frame rate from 30 FPS to 60 FPS, and its mounting dimensions differ.

|       Item        | Quantity |                        Specification                         |                           Remarks                            |
| :---------------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Stereo RGB Camera |    1     | [60FPS, 125°FOV, 60mm baseline](https://e.tb.cn/h.S2zMNwiUzC9I2H1) |                  For robot head perspective                  |
|  M4x16mm Screws   |    2     |           [Reference](https://amzn.asia/d/cfta55x)           |                 For fastening camera bracket                 |
| M2x5mm/6mm Screws |    8     |           [Reference](https://amzn.asia/d/1msRa5B)           | For fastening (camera - camera bracket) and (camera bracket - camera cover) |

- 3D Printing Parts

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%; text-align: center;">
  <colgroup>
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;">
  </colgroup>
  <tr>
    <th>Item</th>
    <th>Camera Bracket</th>
    <th>Camera Cover Plate</th>
    <th>USB-Type-C Clamp</th>
    <th>Download Link</th>
  </tr>
  <tr>
    <td>
      <br />
      <b>Classic Head (98mm)</b>
    </td>
    <td></td>
    <td></td>
    <td align="center"></td>
    <td>📥 Classic 3D Printing Parts</td>
  </tr>
  <tr>
    <td>
      <br />
      <b>Renewed Head (88mm)</b>
    </td>
    <td></td>
    <td></td>
    <td align="center"></td>
    <td>📥 Renewed 3D Printing Parts</td>
  </tr>
</table>

### 5.2.2 Stereo Camera 30 FPS

- Materials

|       Item        | Quantity |                        Specification                         |                           Remarks                            |
| :---------------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   Stereo Camera   |    1     | [30FPS, 125°FOV, 60mm baseline](http://e.tb.cn/h.TaZxgkpfWkNCakg) |                  For robot head perspective                  |
|  M4x16mm Screws   |    2     |           [Reference](https://amzn.asia/d/cfta55x)           |                 For fastening camera bracket                 |
| M2x5mm/6mm Screws |    8     |           [Reference](https://amzn.asia/d/1msRa5B)           | For fastening (camera - camera bracket) and (camera bracket - camera cover) |

- 3D Printing Parts

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%; text-align: center;">
  <colgroup>
    <col style="width: 20%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 30%;">
  </colgroup>
  <tr>
    <th>Item</th>
    <th>Camera Bracket</th>
    <th>Camera Cover Plate</th>
    <th>Download Link</th>
  </tr>
  <tr>
    <td>
      <br />
      <b>Classic Head (98mm)</b>
    </td>
    <td></td>
    <td>None</td>
    <td>📥 Classic 3D Printing Parts</td>
  </tr>
  <tr>
    <td>
      <br />
      <b>Renewed Head (88mm)</b>
    </td>
    <td></td>
    <td></td>
    <td>📥 Renewed 3D Printing Parts</td>
  </tr>
</table>

### 5.2.3 G1 Wrist RealSense D405

> RealSense D405 is recommended only for [Dex3-1]() end-effector use.

- Materials

|      Item      | Quantity |                        Specification                         |                           Remarks                            |
| :------------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| RealSense D405 |    2     | [Website](https://www.intelrealsense.com/depth-camera-d405/) | For G1 robot wrist (M4010 motors) left & right perspectives  |
|   USB3.0 Hub   |    1     | [Issue](https://github.com/IntelRealSense/librealsense/issues/24) | Choose a high-quality hub; recommended to connect to [Type-C #9]() |
|  M3-1 Hex Nut  |    4     |             [Reference](https://a.co/d/gQaLtHD)              |                     For wrist fastening                      |
|  M3x12 Screw   |    4     |           [Reference](https://amzn.asia/d/aU9NHSf)           |                     For wrist fastening                      |
|   M3x6 Screw   |    4     |           [Reference](https://amzn.asia/d/0nEz5dJ)           |                     For wrist fastening                      |

- 3D Printing Parts

|           Item           | Quantity |            Remarks             |                        Download Link                         |
| :----------------------: | :------: | :----------------------------: | :----------------------------------------------------------: |
|     D405 Wrist Ring      |    2     |  To be used with wrist bracket   | [📥 STEP]() |
| Left Wrist Camera Bracket  |    1     | For mounting left D405 camera  | [📥 STEP]() |
| Right Wrist Camera Bracket |    1     | For mounting right D405 camera | [📥 STEP]() |

### 5.2.4 G1 Wrist Monocular Camera

- Materials

|       Item        | Quantity |                        Specification                         |                      Remarks                       |
| :---------------: | :------: | :----------------------------------------------------------: | :------------------------------------------------: |
| Monocular Camera  |    2     | [60FPS, 160° FOV](https://e.tb.cn/h.S2YWUJan6ZP8Wqv?tk=MqHK4uvWlLk) |   For G1 robot wrist (M4010 motors) left & right   |
|    USB3.0 Hub     |    1     | [Reference](https://e.tb.cn/h.S2QB8hVuKbNfqb9?tk=XsBL4uwn2Ch) |          For connecting two wrist cameras          |
|   M3-1 Hex Nut    |    4     |             [Reference](https://a.co/d/gQaLtHD)              |                For wrist fastening                 |
|    M3x12 Screw    |    4     |           [Reference](https://amzn.asia/d/aU9NHSf)           |         For fastening wrist bracket and ring         |
|   M2.5x5 Screw    |    4     |           [Reference](https://amzn.asia/d/0nEz5dJ)           |     For fastening cable clip and wrist bracket     |
| M2x5mm/6mm Screws |    8     |           [Reference](https://amzn.asia/d/1msRa5B)           | For fastening (camera-bracket) and (bracket-cover) |

- 3D Printing Parts

<table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr>
    <th>End-Effector</th>
    <th>Camera Bracket</th>
    <th>Wrist Ring</th>
    <th>Camera Cover Plate</th>
    <th>Cable Clip</th>
    <th>Download Link</th>
  </tr>
  <tr>
    <td>Dex1-1</td>
    <td></td>
    <td></td>
    <td></td>
    <td rowspan="3" valign="middle">
      
    </td>
    <td rowspan="3" valign="middle">
      📥 Download 3D Printing Parts
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


## 5.3 🔨 Installation Illustrations (Partial)

<table>
    <tr>
        <th align="center">Item</th>
        <th align="center" colspan="2">Simulation</th>
        <th align="center" colspan="2">Real Device</th>
    </tr>
    <tr>
        <td align="center">Head</td>
        <td align="center">
            <p align="center">
                <img src="./img/head_camera_mount.png" alt="head" width="100%">
                <figcaption>Head Bracket</figcaption>
            </p>
        </td>
        <td align="center">
            <p align="center">
                <img src="./img/head_camera_mount_install.png" alt="head" width="80%">
                <figcaption>Assembly Side View</figcaption>
            </p>
        </td>
        <td align="center" colspan="2">
            <p align="center">
                <img src="./img/real_head.jpg" alt="head" width="20%">
                <figcaption>Assembly Front View</figcaption>
            </p>
        </td>
    </tr>
    <tr>
        <td align="center">Wrist</td>
        <td align="center" colspan="2">
            <p align="center">
                <img src="./img/wrist_and_ring_mount.png" alt="wrist" width="100%">
                <figcaption>Wrist Ring & Camera bracket</figcaption>
            </p>
        </td>
        <td align="center">
            <p align="center">
                <img src="./img/real_left_hand.jpg" alt="wrist" width="50%">
                <figcaption>Assembly Right Hand</figcaption>
            </p>
        </td>
        <td align="center">
            <p align="center">
                <img src="./img/real_right_hand.jpg" alt="wrist" width="50%">
                <figcaption>Assembly Left Hand</figcaption>
            </p>
        </td>
    </tr>
</table>


> Note: As shown in the red circles, the wrist ring bracket must align with the wrist joint seam.
