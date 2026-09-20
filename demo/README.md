# 抓取 Demo 代码分析文档索引

这一套文档把 `carm_grasp-main/examples/benchmark/src/` 下两个**真正执行抓取**的脚本，按"零基础也能看懂"的标准逐行讲了一遍。

---

## 两篇文档

| 文档 | 源文件 | 干什么 | 需要什么输入 |
|---|---|---|---|
| [`test_tmpl_grasp_2d/test_tmpl_grasp_2d_analysis.md`](test_tmpl_grasp_2d/test_tmpl_grasp_2d_analysis.md) | `test_tmpl_grasp_2d.py`（711 行） | **2D 视觉伺服抓取**：物体贴在水平桌面上，只管 `x, y, θ` 三个自由度，靠"图像误差 → 末端增量"反复逼近 | 只有 RGB 图（**不需要深度**）+ 5 个模板状态 |
| [`test_tmpl_grasp_3d/test_tmpl_grasp_3d_analysis.md`](test_tmpl_grasp_3d/test_tmpl_grasp_3d_analysis.md) | `test_tmpl_grasp_3d.py`（857 行） | **3D 六自由度抓取**：物体随便摆，6 个自由度全解算，先到"预备位姿"再走相对增量到抓取位姿 | RGB-D 图 + 2 个模板位姿 + 夹爪标定文件 |

---

## 2D 和 3D 到底差在哪

| 维度 | 2D 抓取 | 3D 抓取 |
|---|---|---|
| 物体自由度 | 3（x, y, θ） | 6（x, y, z, roll, pitch, yaw） |
| 相机 | 单目 RGB | RGB-D |
| 视觉输出 | `pose_2d = [nx, ny, θ]`（归一化坐标 + 朝向） | `T_cam_model`（4×4 位姿矩阵） |
| 要不要知道物体多远 | 不需要，靠 near/far 两次示教"标定"出比例 | 需要，深度图直接给 |
| 控制方式 | **闭环伺服**：看一眼 → 动一点 → 再看一眼，最多 20 次 | **开环 + 细化**：算一次目标位姿 → 移动 → 最多再细化 2 次 |
| 模板内容 | 5 个状态（grasp / near / next_near / far / next_far） | 2 个状态（grasp / ready） |
| 适用场景 | 平面上的物体、相机朝下、要求不高 | 任意姿态的物体 |

**一句话选型**：物体平躺在桌上、相机垂直朝下 → 用 2D，便宜且稳；物体姿态乱七八糟 → 必须用 3D。

---

## 跑之前必须完成的前置标定

```
① calib_camera.py    → cam_params.json     （fx,fy,cx,cy + 畸变）
② calib_handeye.py   → calib_handeye.json  （T_end_cam）
③ calib_gripper.py   → gripper.json        （T_cam_gripper）   ← 只有 3D 需要
④ create_tmpl_grasp_2d.py / create_tmpl_grasp_3d.py
                     → tmpl_dir/           （示教出来的模板）
        ↓
   test_tmpl_grasp_2d.py / test_tmpl_grasp_3d.py
```

> ⚠️ 模板是**示教**出来的，不是算出来的。`create_tmpl_grasp_*.py` 用键盘把机械臂摆到各个位置，按 `g` / `n` / `b` / `f` / `d` 存下当时的位姿和图像观测。**换物体、换相机、动过相机都必须重新示教**。

---

## 两个脚本共同的双线程架构

```
主线程                              子线程
─────────                           ─────────
while rclpy.ok() and not stop:      run()
    rclpy.spin_once(cam_node, 0.1)      ├─ 移到检测位姿
        │                               ├─ do_grasp()
        │ 回调写 cam_node.imgs          │    ├─ 取帧 cam_node.get_frames()
        └──────────────────────→        │    ├─ 视觉匹配 / 跟踪
                                        │    ├─ 算目标位姿
                                        │    └─ arm.set_pose()（阻塞）
                                        ├─ 移到放置位姿
                                        └─ stop_event.set()  ← 通知主线程退出
```

**为什么要这么分**：`arm.set_pose()` 是阻塞调用（机械臂在动的几十毫秒到几秒里，代码卡在那一行）。如果单线程，这段时间没人 `spin`，相机回调不触发，图像永远是旧的。拆成两个线程后，主线程专职收图，子线程专职控制。

**关键契约**：子线程里 `get_frames()` 必须用 `do_spin_once=False`（默认），否则两个线程同时 spin 同一个节点会打架。见 `cam_ros_utils` 文档第 7 节。

---

## 通用排查顺序

抓取不准时，按这个顺序查（**八成不是代码的问题**）：

```
1. 先在 rviz 里订阅 /target_arm_pose，看算出来的目标位姿是不是你想要的
   （这一步能区分"视觉错了"还是"控制错了"）
2. 打开 debug 模式（--debug），一步步按回车，看每步的日志数值
3. 检查 debug_dir 里存的图：results/debug/grasp_2d/ 或 grasp_3d/
   - match/color.png      看 tag 认出来没有
   - track/proj_area.png  看跟踪的裁剪框框对没有
4. 核对单位：日志里的平移是"米"还是"毫米"？深度图有没有乘 depth_scale？
5. 核对标定文件是不是这次刚生成的（最容易忘）
6. 重新做模板：示教时的位姿和抓取时的位姿差太远会发散
```

---

## 两个脚本共有的坑（跨文档通用）

| 坑 | 现象 | 说明 |
|---|---|---|
| `eye_in_hand` 标志被丢弃 | 喂了"眼在外"的标定文件也不报错 | 两处都是 `T_end_cam, _ = read_calib_handeye(...)`，矩阵被当成 `T_end_cam` 用 |
| 只 spin 相机节点 | rviz 里夹爪看不见 | `TargetArmNode()` 默认 `pub_gripper_msg=False` |
| 主线程 Ctrl+C 有时失效 | 要按好几次 | 子线程的 `wait_key()` 用 `input()` 阻塞 + `KeyboardReader` 的 cbreak 模式吞掉了 SIGINT |
| 退出时机械臂有时不动 | 日志报 disconnect 失败 | `run()` 里 `arm.set_joints()` 和主线程 finally 里重复回零 |
| 调试图颜色发蓝/发红 | 存的 png 颜色不对 | cv_bridge 按 `rgb8` 解码成 RGB，但 OpenCV 的 `imwrite`/画线按 BGR 约定 |

---

## 源文件位置

```
C:\Users\x\Learn\grasp\robot_grasp\carm_grasp-main\examples\benchmark\src\
├── create_tmpl_grasp_2d.py   ← 2D 模板示教（本文档暂未覆盖，需要可补）
├── create_tmpl_grasp_3d.py   ← 3D 模板示教（本文档暂未覆盖，需要可补）
├── test_tmpl_grasp_2d.py     ← ✅ 已分析（711 行）
└── test_tmpl_grasp_3d.py     ← ✅ 已分析（857 行）
```

它们依赖的工程模块见 [../core/README.md](../core/README.md)。

> ⚠️ `apriltag2` 不在工程目录内，是外部依赖（需另行安装），两个脚本都靠它做 AprilTag 检测。
