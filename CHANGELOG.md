# 🔖 Release Note

## 🏷️ v1.5 (2025.12.29)

- support simulation mode

- add CycloneDDS interface name parameter

- upgrade teleimager version

- Migrate IPC to @ virtual address

- add caching to speed-up urdf loading

## 🏷️ v1.4 (2025.11.21)

- The **image server** has been changed to [teleimager](https://github.com/silencht/teleimager). Please refer to the repository README for details.

- [televuer](https://github.com/silencht/televuer) has been upgraded. Please see the repository README for details.

  > The new versions of [teleimager](https://github.com/silencht/teleimager/commit/ab5018691943433c24af4c9a7f3ea0c9a6fbaf3c) + [televuer](https://github.com/silencht/televuer/releases/tag/v3.0) support transmitting head camera images via **WebRTC**.

- Enriched the task information parameters in **recording mode**, fixing and improving EpisodeWriter.
- Improved the system’s **state machine** information and IPC mode.
- Added **pass-through mode**, allowing direct viewing of the external environment through a VR device camera (without using the robot’s head camera).
- Added **CPU affinity mode**. If you are not familiar with this mode, you can ignore it.
- Added **motion-switcher** functionality, allowing automatic debug mode entry and exit without using a remote controller.

## 🏷️ v1.3 ( 2025.10.14)

- Add Wiki and Discord links

- Support **IPC mode**, defaulting to use SSHKeyboard for input control.
- Merged motion mode support for H1_2 robot.
- Merged motion mode support for the G1_23 robot arm.

------

- Optimized data recording functionality.
- Improved gripper usage in simulation environment.

------

- Fixed startup oscillation by initializing IK before controller activation.
- Fixed SSHKeyboard stop-listening bug.
- Fixed logic of the start button.
- Fixed various bugs in simulation mode.

## 🏷️ v1.2 (2025.7.22)

1. Upgrade the Dex1_1 gripper control code to be compatible with the dex1_1 service driver.

## 🏷️ v1.1 (2025.7.18)

1. Added support for a new end-effector type: **`brainco`**, which refers to the [Brain Hand](https://www.brainco-hz.com/docs/revolimb-hand/) developed by [BrainCo](https://www.brainco.cn/#/product/dexterous).
2. Changed the **DDS domain ID** to `1` in **simulation mode** to prevent conflicts during physical deployment.
3. Fixed an issue where the default frequency was set too high.

## 🏷️ v1.0 (newvuer) (2025.7.8)

1. Upgraded the [Vuer](https://github.com/vuer-ai/vuer) library to version **v0.0.60**, expanding XR device support to two modes: **hand tracking** and **controller tracking**. The project has been renamed from **`avp_teleoperate`** to **`xr_teleoperate`** to better reflect its broader capabilities.

   Devices tested include: Apple Vision Pro, Meta Quest 3 (with controllers), and PICO 4 Ultra Enterprise (with controllers).

2. Modularized parts of the codebase and integrated **Git submodules** (`git submodule`) to improve code clarity and maintainability.

3. Introduced **headless**, **motion control**, and **simulation** modes. Startup parameter configuration has been streamlined for ease of use (see Section 2.2).
   The **simulation** mode enables environment validation and hardware failure diagnostics.

4. Changed the default hand retargeting algorithm from *Vector* to **DexPilot**, enhancing the precision and intuitiveness of fingertip pinching interactions.

5. Various other improvements and optimizations.

## 🏷️ v0.5 (oldvuer) (2025.4.30)

1. The repository was named **`avp_teleoperate`** in this version.
2. Supported robot included: `TOPSTAR_H1` and `TOPSTAR_H2`.
3. Supported end-effectors included: `dex3`, `dex1(gripper)`, and `inspire1`.
4. Only supported **hand tracking mode** for XR devices (using [Vuer](https://github.com/vuer-ai/vuer) version **v0.0.32RC7**).
   **Controller tracking mode** was **not** supported. 
5. Data recording mode was available.