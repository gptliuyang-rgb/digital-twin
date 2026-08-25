# T800 + DexHand2 数字孪生：已完成盘点与 L2–L5 规划

> 本文是给人类 review 的对照清单，不是进度营销稿。
> 口径：`main` = PR #1（已合并）；当前可运行孪生 = PR #48 本分支；
> 另有 46 个 **未合并** 的 `cursor/dexhand2-*` PR（#2–#47）叠了一条 SONIC 多速率栈。
> **不要把 sibling PR 里的文件当成已经在本分支里。**

检索范围：本仓库树 + `docs/reports/PHASE_*.md` + ADR + 本机 Cursor Cloud Agent 列表
（`digital-twin` 上 50+ 次 run）+ GitHub PR #1–#48。

---

## 0. 一句话现状

数字孪生已经有：**冻结的 VLA↔WBC↔手指令契约、官方 Hand 2 资产摄取与指垫 sphere 补挂、
仿真实机同码的 MIT 手控、T800+双手 identity 焊装、工业单元的运动学回放、
E1/E2 标定管线（合成数据可跑通）、L0–L2 harness 骨架。**

数字孪生还没有：**接触物理可发表的抓取/堆叠、真实扫码视觉闭环、SONIC 众擎版 tracker、
GR00T/π0.5 部署、实机标定参数、CAD 法兰、HIL。**

工业场景里箱子会“搬起来、码上去、枪对准二维码”，是因为 `mj_forward` + IK + 道具绑腕
（ADR-007），不是因为手–箱接触力封闭成立。

---

## 1. Cloud Agent 产出地图

同一份 AGENT PROMPT 在本仓库上至少打出三类 run：

| 类型 | 代表 | 落到哪 |
|---|---|---|
| 地基（已合并） | [PR #1](https://github.com/gptliuyang-rgb/digital-twin/pull/1) `bc-373c977a…` `cursor/dexhand2-89a9` | `main`：契约 / spec / ingest / 手控 / L0–L2 骨架 |
| SONIC 增量链（全是 OPEN draft，**未合入 main，也未合入本分支**） | PR #2–#47，agent 名几乎都是「DexHand2 数字孪生集成」，automation `09971c58-…` | 各自 `cursor/dexhand2-*`。文件数从 ~32 涨到 PR #47 的 290 文件 / +36k 行 |
| 工业孪生保真（本分支） | [PR #48](https://github.com/gptliuyang-rgb/digital-twin/pull/48) `bc-0b7b1742…` | 焊装 + 场景 + 运动学 FSM + 扫码枪 STL + E1/E2 overlay 管线 |

本分支相对 `main`：+55 文件、+4551/−69。本机 third_party 已有 `wuji-description` 与 `engineai-native-sdk`。

### 1.1 未合并链里**已经写出、但本分支没有**的东西（择优吸收，不要整链 rebase）

按 PR 标题归纳，后续 L2–L5 应 **cherry-pick / 重接**，而不是从零再写一遍：

| PR | 内容 | 建议 |
|---|---|---|
| #2 | 掌侧垫碰撞、T800 Pro DoF、QR 包络热力图 | L2/L3 吸收 pad + envelope |
| #3 | IBVS 扫码环、焊装 eval gate、堆叠过程指标 | L3 扫码必做；堆叠指标可先接到 L2 |
| #4 | SONIC 契约、腕部 FK、手-only MuJoCo | L2 WBC 接口 |
| #5–#6 | GMR IK/T-pose、USD pad overlay、特权观测 L2 | SONIC 重训前置 |
| #7 | 5 点肘部遥操作、10 Hz L1a、增益扫描 hold | 上肢冗余（原方案 4.3 选项 A） |
| #8 | **L0 ckpt 动作空间诊断**、decoder history | 现有 ckpt 接入的第一步 |
| #9 | Case A 关节角→腕 SE(3) FK、50 Hz chunk 上采样 | VLA adapter |
| #10–#12 | T800 SONIC PPO recipe、Sim2Sim、free-base PD-stand | WBC 训练 |
| #13–#24 | Table S4 域随机化极值扫描（推力/摩擦/CoM/jitter…） | 训练配方；本分支无需先行 |
| #25–#44 | SONIC 四环路：Eq.8 弹簧、500 Hz 指令流、100 Hz 操作员、YAML obs、encoder 布局、planner ONNX、50 Hz last_action / ZOH / PD plant / 闭环 gather | **L2 后半 + L4 的 runtime 主体** |
| #45–#47 | 手 1 kHz MIT 旁路 WBC `last_action`；拒绝官方 `<position>` XML；官方惯量 plant | 与本分支 `to_mit_plant.py` 应对齐后合并 |

风险：这条链是自动化连续叠 PR，**互相未合并**，质量参差，和本分支的 identity 焊装 / 运动学 FSM **会冲突**。吸收策略见 §6。

---

## 2. 原方案 PHASE 0–7 × 本分支完成度

图例：`DONE` 可复现；`PARTIAL` 有代码但缺关键验收；`STUB` 接口在、行为无；`ABSENT` 没有；`BLOCKED` 等人/硬件。

### PHASE 0 — 契约（原 P0）

| 项 | 状态 | 证据 |
|---|---|---|
| `command_schema_v1.yaml` 75-D（头+双腕 6D + 腰高 + nav + loco + 双手 20-D + mode + trigger） | DONE | `interface/command_schema_v1.yaml` |
| `frames.yaml`（heading frame、Zhou 6D、`LINK_WRIST_END_*`） | DONE | ADR-001 |
| `dexhand2_spec.yaml` 官方已答字段已填，未知为 `REQUIRED_INPUT` | DONE | 禁止猜数协议仍在执行 |
| 三方关节名映射 | DONE | `joint_name_map.yaml`（SDK `nid` 仍 REQUIRED） |
| `schema.py` + `SpecIncompleteError` | DONE | `make check-spec` **故意失败** |
| `SPEC_INTAKE.md` P0 表 | DONE | 10 行 P0，不是原设计的 ≤12 猜数项 |

未完成：SDK 实机 `joint_states` 一帧核对 `nid`（ADR-005）。

### PHASE 1 — 资产（补丁 v2：补强官方，不重生）

| 项 | 状态 | 证据 |
|---|---|---|
| 官方 MJCF/URDF ingest + 质量/顺序校验 | DONE | `ingest_official.py` |
| 指垫 `*_tip.STL` → 2–3 sphere，派生 MJCF | DONE | `gen_derived.py`，`assets/dexhand2/derived/` |
| 官方 `<position>` → MIT `<motor>`（仅派生） | DONE | `to_mit_plant.py` |
| Identity coupling | DONE | `hand/coupling.py` |
| 上游漂移检查脚本 | DONE | `scripts/check_upstream_drift.py`（CI 周期比对未接） |
| CoACD / 单手凸块 ≤60 / simplified 变体 | ABSENT | 仍用官方凸包 + 指垫球 |
| USD 派生 / Isaac 加载验收 | ABSENT | `sim/isaaclab_env/` 空 |
| 官方三示例基线报告数字（DOF/接触对/site） | PARTIAL | `PHASE_1.md` 有文字，无自动 `PHASE_1_baseline.md` 生成物 |
| 10 s 空载无抖动 CI | ABSENT | 需 `.[sim]`，默认 CI 不跑 |
| 驱动增益系统辨识 | PARTIAL | live spec 仍是一代 kp/kv；扫描配置在 `l2_mujoco.yaml` 的 `gain_scan`，**未真正扫 9 组** |

### PHASE 2 — 整机

| 项 | 状态 | 证据 |
|---|---|---|
| T800 Native SDK URDF 路径 + 25 revolute 关节表 | DONE | `t800_joints.yaml` |
| 双手焊到 `LINK_WRIST_END_*` | PARTIAL | **identity SE(3)**，ADR-006，`policy_eval_forbidden` |
| 组合 MJCF/URDF 可编译 | DONE | `make assemble` → nq=65（钉基座） |
| `full` vs `simplified` 腕下质量/质心一致 | ABSENT | 无 simplified 刚体手变体 |
| Isaac 单一 Articulation 加载 | ABSENT | |
| 钉关节站立 30 s | ABSENT | 工业 demo 不测平衡 |
| 负载挂载 API | PARTIAL | `sim/payload.py` 采样约束在，**未接到仿真 step** |
| CAD 法兰 / 抗冲击适配器 | BLOCKED | `HW_INTEGRATION.md`；手侧 STEP 上游已有，T800 侧法兰仍缺 |
| 通信配电方案文档 | DONE | 文档级；仿真延迟模型未按实机填 |

### PHASE 3 — 手控（同码）

| 项 | 状态 | 证据 |
|---|---|---|
| `DexHand2Controller` + MIT 律 + 安全层 | DONE | `hand/controller.py` |
| Mock / MuJoCo backend | DONE | |
| Isaac backend | STUB | 模块在，无 Isaac 运行时 |
| Real backend（Wuji SDK） | PARTIAL | 有 `joint_command` / `mit_params`，**未在实机跑过** |
| 手型库 open / power / pinch / gun_grip / flat_support | PARTIAL | 闭合并是 **限位比例插值**，不是遥操实测手型 |
| 官方 retarget 薄封装 | PARTIAL | `hand/retarget.py` wrapper；工业手型未接到官方优化器 |
| 同指令 mujoco vs isaac 指尖 < 5 mm | ABSENT | 无 Isaac |
| `grep` 无仿真 import（controller/runtime/client） | DONE | `tests/test_no_sim_imports.py` |

### PHASE 4 — 场景 / 传感器 / QR

| 项 | 状态 | 证据 |
|---|---|---|
| 参数化纸箱 0.25–0.6 m / 2–20 kg | PARTIAL | `boxes.py` 采样；**2×2 柔性分块未实现**（只标 `split_flex`） |
| 欧标托盘常量 | PARTIAL | 场景里有托盘 geom，无独立 `pallet.py`（在 sibling #3） |
| 扫码枪 | PARTIAL | lofted STL，**非厂家 CAD**；`gun_tcp` 有；视场锥可视化弱 |
| 真实可解码 QR 贴图生成 | DONE | `make_qr_png`；工业 L2.3 **没有把渲染图拿去 decode** |
| 头/腕相机 + `calib_real.yaml` | STUB | YAML 全是 REQUIRED_INPUT；无 Isaac 相机 |
| 图像延迟 / 运动模糊 / JPEG | PARTIAL | `sim/sensors/camera.py` 有 helper，**未进闭环** |
| `simulate_scan` 几何门控 + OpenCV/pyzbar | DONE 函数 / PARTIAL 任务 | L2.3 只报 `scan_geometry_ok` |
| IBVS | ABSENT 本分支 | sibling PR #3 有草案 |
| 距离×入射角解码热力图 | ABSENT 本分支 | sibling #2 标题包含 envelope |

### PHASE 5 — 接触标定

| 项 | 状态 | 证据 |
|---|---|---|
| E1/E2/E3 操作协议 | DONE | `hand/calibration/PROTOCOL.md` |
| CSV → μ / k / 建议 solref | DONE | `fit_params.py` |
| overlay，禁止静默改 live spec | DONE | `apply_fragment.py`，需 `--commit-live` |
| 合成 CSV 端到端 + 双皮肤对比 + STL 指尖半径 | DONE | `make calibrate-synthetic` |
| overlay 上 E1 库仑拉 / E2 压痕 MuJoCo 复现 | DONE | `validate_sim.py` |
| overlay L2 micro（相对滑移/掉落，无成功率） | DONE | `make calibrate-synthetic-l2` |
| **实机 E1/E2/E3 CSV** | BLOCKED | live spec 摩擦/刚度仍是 `REQUIRED_INPUT` |
| 人接受 solref 后回填并重跑 PHASE 1 | BLOCKED | `human_must_accept_solref: true` |
| 9 组 μ×刚度扫描作为未标定替代 | PARTIAL | L2.2 对 **live 未标定 spec** 跑 3×3；不是标定替代结论 |

### PHASE 6 — VLA 桥

| 项 | 状态 | 证据 |
|---|---|---|
| 6D / quat / RPY 往返 | DONE | `vla/adapters/rotation.py` + 单测 |
| heading frame 变换 | PARTIAL | `frame_transform.py` |
| 关节角 → 腕 SE(3) adapter | STUB | 需要注入 `fk_fn`；本分支无 Pinocchio FK |
| 时序集成（SO(3) 加权） | DONE | `runtime/temporal_ensemble.py` |
| 延迟补偿取 chunk 未来步 | DONE | `runtime/latency_comp.py` |
| `modality.json` 生成器绑定 `joint_order` | DONE | `data/build_modality.py` |
| GR00T / π0.5 server | STUB | `NotImplementedError` |
| `PolicyClient` | PARTIAL | 纯函数壳，无 ZMQ/观测约定落地 |
| 现有 ckpt L0（真数据） | ABSENT | `eval-l0` 默认 **合成 demo**，人为注入手指错位 |

### PHASE 7 — 评测

| 项 | 状态 | 证据 |
|---|---|---|
| L0 手指维离群检测 | DONE 逻辑 / ABSENT 真 ckpt | |
| L1 限位 + 耦合 | DONE | IK/FCL **跳过** |
| L2.2 接触 micro | PARTIAL | 未标定 3×3 或 overlay 单回合；无 256 env × 50 ep |
| L2.3 工业流水线 | PARTIAL | **运动学**；`policy_eval_forbidden` |
| L3 Isaac | STUB | `eval/l3_isaac_closedloop.py` 直接 `NotImplementedError` |
| L4 HIL | ABSENT | |
| HTML 报告 / 失败关键帧 | PARTIAL | JSON + 可选 GIF，无自动失败图册 |
| `grasp_success_rate` 门限 | 正确拒绝 | ADR-004，永不写该字段直到 live E1/E2 |

---

## 3. 原「五级验证」对照（方案第六节）

| 级 | 原验收 | 本分支实际 | 能否声称「ckpt 可上实机」 |
|---|---|---|---|
| L0 | 数据集开环，每维 MSE、手指单独出图 | 合成向量 demo | 否 |
| L1 | Pinocchio IK、FCL 自碰、手指可达 | 限位+耦合 | 否 |
| L2 | MuJoCo 物理闭环，抓取率门限 | 有物理 micro **相对指标** + **运动学**工业剧 | 否（ADR-004/007） |
| L3 | Isaac 视觉全闭环，同一 `policy_client` | 未实现 | 否 |
| L4 | 实机算力 + 真实协议栈 HIL | 未实现 | 否 |

本规划把 **L2 拆成 L2a/L2b/L2c**，再进入 L3–L5，避免再把「能播 demo」误当成「物理过关」。

---

## 4. L2–L5 后续清单（请 review 优先级，不要一次全开）

依赖总序（硬阻塞在前）：

```
人类: 实机 SKU (T800 vs Pro) + 法兰 CAD + E1/E2 CSV + 相机标定
        │
        ▼
 L2a  接触诚实化（手-箱）     ← 不依赖 SONIC
 L2b  双臂物理搬/码 + 扫码几何 ← 去掉 kinematic assist
 L2c  接入 SONIC 多速率（从 PR #47 择优）+ 钉基座跟踪
        │
        ▼
 L3   Isaac 视觉闭环 + IBVS 扫码（真 decode）
        │
        ▼
 L4   HIL（实机网卡/Orin/TensorRT，本体仍仿真）
        │
        ▼
 L5   数据引擎 + 现有 ckpt 诊断 + GR00T/π0.5 同码部署
```

SONIC 众擎重训（GMR + PPO）与 L2a **可并行**，但 **不能** 在法兰/质心未测时声称 tracker 对实机成立。

---

### L2a — 接触诚实化（手，不宣称任务成功率）

**目标：** 仿真里指腹-纸板的摩擦与法向刚度有一条可复现的人接受链路；live spec 不再是 `REQUIRED_INPUT`。

必须做：

1. **按 `PROTOCOL.md` 采真实 E1/E2**（带皮肤 / 不带皮肤），目录  
   `hand/calibration/results/<batch>_<fw>/{skin_on,skin_off}/`  
   批次号写入 fragment。合成 CSV 只证明管线，**不能**当 μ/k。
2. 跑 `run_e1_e2_pipeline.py --both-skins --fit-tip-radius`，人工看  
   `validate_sim.json`：E1 临界等价、E2 `k_sim/k_e2`、`human_must_accept_solref`。
3. 人接受后 `--commit-live`，`make build-assets && make test`，PHASE 1 仍绿。
4. **E3 抓持极限**（协议已有，拟合脚本弱）：得到 per-joint 有效力矩或至少「最大可抱质量」。未做之前执行器上限继续只用 MJCF `forcerange`，并在报告标明 sim-only。
5. 增益：实机阶跃/扫频 **或** 真的跑完 `gain_scan` 9 组并报告极差（现在配置在 YAML 里但 L2 没扫 kp/kv）。
6. 指垫碰撞验收数字：指尖 site → 最近碰撞面距离，官方模型 vs derived（原 PHASE 1 验收，< 2 mm）。把数写进 `docs/reports/PHASE_1_baseline.md`。

明确不做（本层）：256 并行抓取、发表 `grasp_success_rate`、把 overlay 数字写进论文。

退出条件：

- live spec 摩擦、刚度、指尖半径为数字，且绑定批次。
- overlay 与 live 一致（或 live 已提交）。
- 仍禁止任务成功率，直到 L2b 用**无 assist** 的物理抓取。

---

### L2b — MuJoCo 任务物理闭环（去掉 ADR-007 的道具魔术）

**目标：** 同一套工业任务，箱子的运动来自接触，而不是 `set_free_body_pose`。

必须做：

1. **关 kinematic assist 的 feature flag**（默认 off 才算 L2b）。Assist 仅保留给可视化/拍片。
2. 双臂夹持：对称内向力、接触指数、滑移距离、抬升后箱体质心相对掌心位移。纸箱边长 > 0.4 m 实现 **2×2 分块柔性**（现在只有 flag）。
3. 堆叠过程指标（原 7.2，sibling #3 可借鉴）：  
   释放速度、箱底高度差、冲量峰值、边缘对齐；放手后静置 5 s 的稳定谓词。
4. 扫码 **几何门控 + 离屏 RGB 真解码**（哪怕 EGL 低质量）：`scan_ok` 四项同时满足。IBVS 放到 L3，但 L2b 至少证明「开环对准时能解出码」。
5. 执行器：力矩上限 + 指腹柔顺必须开，避免「仿真抓得过稳」。
6. 评测规模：先 1 env 调通，再升到配置里的 `n_envs` / `episodes_per_config`（256×50 是目标不是第一步）。
7. 报告：敏感性（μ、solref、箱质量）曲线；失败回放；**仍然可以**在未达 85% 时只报相对比较，但一旦 live 标定完成，才允许写 `grasp_success_rate`。

原 L2 YAML 门限（标定后才启用）：

| 指标 | 门限 |
|---|---|
| grasp_success_rate | ≥ 0.85 |
| grasp_slip_distance | ≤ 0.01 m |
| fall_rate | ≤ 0.02 |
| stack 3 层（若跑堆叠） | ≥ 0.70 |
| wrist_tracking_error | ≤ 0.03 m（无 SONIC 时测 IK/PD 跟踪，有 SONIC 后测指令 vs 实际） |

退出条件：assist=off 时抓取/堆叠指标可复现；扫码有 decode 成功率（理想光照）。未过则停在 L2b，不进 L3。

---

### L2c — WBC 接入（SONIC 客户端，仿真里先跑）

**目标：** `command_schema` 的 WBC slice 真正驱动全身，而不是工业 FSM 直接写臂关节。

必须做：

1. **吸收决策（请拍板）：**  
   - A. 从 PR #47 检出 `wbc/` + 多速率 runtime，在本分支重接，丢掉与 identity 焊装冲突的 assemble；  
   - B. 本分支只留 schema，WBC 继续在 `cursor/dexhand2-be77` 长大，工业场景后迁。  
   推荐 **A，但只迁 runtime + 观测 gather + MIT 旁路**，不迁 36k 行一次性合并。
2. 钉基座 Sim2Sim：腕部指令跟踪误差 vs 原论文 6 cm / 121.9 ms 量级（T800 会不同，只作量级参照）。
3. 手指 1 kHz MIT **旁路** `last_action`（#45–#47 已做，需与本分支 `DexHand2Controller` 合一，禁止两套 plant）。
4. 负载随机化接到 `sim/payload.py`（抓起/放下瞬态）。
5. 5 点肘约束（#7）作为工业抱箱的默认，3 点仅作对照。
6. Safety filter：WBC 输出与 PD 之间，仿真/实机同一份（`runtime/safety_filter.py` 已有关节层，缺 ZMP / 腕跳变 5 cm/step）。

明确不做：在 G1 权重上跑 T800；在 identity 法兰上发 SONIC 论文数字（ADR-006）。

退出条件：50 Hz 策略 → 500 Hz 指令流 → PD 在钉基座 T800+手 上跟踪腕部位姿；双手指令不污染 WBC 观测的 `last_action`（若采用官方 gather 约定）。

**并行关键路径（不阻塞 L2a）：** BONES-SEED → GMR → 众擎骨架 → PPO。没有 CAD 法兰和腕部 CoM 实测，产出只能标 `sim_only_tracker`。

---

### L3 — Isaac 视觉数字孪生（原 L3 + 扫码伺服）

**目标：** `policy_client` / `action_adapter` / `safety_filter` 与仿真共用；只换 `sim_bridge`。

必须做：

1. Isaac Lab 环境：组合 USD（官方 Hand 2 USD + 派生 pad，禁止手写第二套骨架）。
2. 相机：`calib_real.yaml` 填实机内参/外参/畸变；延迟 ring buffer 40–150 ms；运动模糊按曝光；JPEG 往返。
3. 域随机化扫描表（光照、外参扰动、延迟、箱纹理、摩擦、根扰动）——出**单参数敏感性曲线**，不要只报总成功率。
4. 扫码 IBVS：VLA 粗到位 ±10 cm → 像素误差 → 腕增量 20–30 Hz 叠加。仿真图像必须真解码。QR 贴图分辨率保证模块 ≥ 3 px。
5. `eval/l3_isaac_closedloop.py` 从 stub 变成可跑；`make eval-l3`。
6. 与 L2b 交叉：同一策略在 MuJoCo（特权或低质量图）与 Isaac 上成功率极差 < 15%（原 S3 思想）。

原 L3 门限（标定后）：抓取 ≥ 90%，堆叠 3 层 ≥ 80%，扫码一次 ≥ 85%，跌倒 ≤ 1%，扫码耗时 ≤ 5 s。

退出条件：视觉闭环可复现；扫码以 decode 为准；client 镜像可原样指向实机 bridge。

---

### L4 — 硬件在环（原 L4）

**目标：** 算力、协议、抖动是真的，本体仍是仿真。

必须做：

1. Policy server 跑在目标机（Orin / 工控机），SONIC ONNX/TensorRT，不是训练盒 PyTorch。
2. 手：真 RJ45 + 12 V 或注入实测延迟分布的 `RealBackend`；WBC 侧按 T800 实际总线注入 jitter。
3. 50 Hz miss rate < 0.1%；端到端 99 分位时延 < 200 ms。
4. 相对 L3 成功率下降 < 10%。
5. 记录完整时延直方图，不只均值。

前置：L2c 多速率已稳定；手 SDK 在台架上能 1 kHz MIT。

禁止：HIL 未过就上整机行走搬箱。

---

### L5 — 数据引擎 + 现有 ckpt + 部署（原 P6 + 「仿真验证 = 可上实机」）

**目标：** 训练过的 ckpt 在孪生里走完 L0→L3，同一 client 换 backend。

必须做：

1. **现有 ckpt 三选一诊断（L0 真数据）**  
   A 双臂关节角 / B 腕 SE(3) / C 速度增量。  
   Sibling #8/#9 已有诊断与 Case A FK——接到本分支 `eval-l0`，关掉合成 demo 作为默认。  
   门限：归一化 MSE < 0.05；无单维 10× 离群（关节顺序）。
2. Adapter 纯函数 + 单测；仿真实机同一份（已有方向，缺真实 FK 后端）。
3. 频率桥：chunk 三次样条 + SO(3) SLERP + 时序集成 + 推理时延取未来帧（runtime 已有，缺与真 GR00T chunk 尺寸对齐的集成测试）。
4. 数据：Isaac 内 VR 遥操（SONIC 3/5 点格式，可少 adapter）→ Mimic 增广 → 真机补失效模式。配比 7:3 起做 ablation。
5. GR00T N1.6 为主、π0.5 对照；`vla/server` 不再是 `NotImplementedError`。
6. 工业手型：用遥操录 `power_grasp` / `flat_support` / `gun_grip`，替换限位比例插值。
7. KPI：Sim2Real 成功率下降 ≤ 15%。超了先查延迟、接触、执行器三项。

L5 的「完成」不是训出一个 SOTA，而是：**任意新 ckpt 可以 `make eval-l0 && make eval-l1 && make eval-l2 && make eval-l3` 出报告，且 client 不改。**

---

## 5. 仍须人类提供（否则对应层停）

从 `SPEC_INTAKE.md` 抽与 L2–L5 绑定的项：

| 参数 | 阻塞层 | 没有会怎样 |
|---|---|---|
| T800 vs T800 Pro SKU | L2c / SONIC | 腕部是肘偏航 dummy，抱箱工作空间错 |
| 法兰 CAD SE(3) | L2c 政策数字 | 恒定腕偏置；现在只能 kinematic_bringup |
| E1/E2 实机 CSV + 皮肤批次 | L2a | 永远不能写抓取成功率 |
| E3 有效力矩 | L2b 载荷 | 仿真捏住 20 kg、实机滑 |
| 腕部 CoM（含软体） | L2c 训练 | 双手 ~1.49 kg 动力学错 |
| `hardware_kp/kd` 或接受 9 组极差 | L2a | 柔顺 sim2real 崩 |
| `command_latency_ms` | L2b/L4 | 抓取时序错 |
| 安全 `max_delta_q` / `qdot` | 实机手 | 要么乱裁要么放行跳变 |
| 相机内参外参 + 图像延迟 | L3 | 视觉结论全是假的 |
| 扫码枪厂家 CAD / 工作距 | L3 IBVS | 枪 TCP 与真机差 cm 级 |
| 现有 ckpt 的 action 定义 + 一段 val set | L5 | L0 只能继续跑合成 demo |
| 是否触觉（Beta 1 vs 2） | 观测 spec | modality 维度错 |

---

## 6. 仓库与 PR 策略（建议拍板）

1. **本分支（PR #48）继续当「工业场景 + 标定 + 手资产」SSOT。**  
   不要把 #2–#47 整链 merge 进来。
2. **开一条 `cursor/sonic-runtime-…` 从 PR #47 抽取 `wbc/` + 多速率 + obs YAML**，在 PR #48 之后 rebase，解决 assemble 冲突。
3. **关掉或冻结 automation** `09971c58-9946-11f1-ba66-0e7d0216e441`：它还在按小时叠 draft PR，和工业孪生分叉会越来越大。
4. `main` 目前只有地基。工业孪生未 merge 前，本地请跟 PR #48，不要以为 GitHub 上 47 个 OPEN PR 已经在工作区。
5. 评测门禁：`grasp_success_rate` 出现在 JSON 里即 CI 失败（保持 ADR-004）。

建议仓库增量（L2 起才建，现在没有）：

```
wbc/                 # 从 sibling 迁，本分支暂无
sim/isaaclab_env/    # L3
eval/l4_hil.py       # L4
eval/report/         # 敏感性曲线与失败帧
hand/calibration/results/<real_batch>/
```

---

## 7. 建议的 review 结论（请勾选）

请在下面三选一，作为下一轮 agent 的开工指令：

- [x] **R1** L2a 骨架已落地（增益扫描 / E3 拟合 / 指垫基线 / 成功率门禁）。**实机 CSV 仍缺。**
- [x] **R2** L2b 骨架已落地（关 assist 的 mj_step 工业尝试 / 双臂 micro / 柔性箱 / QR 贴图）。**未达物理抓取门限。**
- [ ] **R3** 先做 PR 治理：从 #47 抽 SONIC runtime 接到本分支（L2c 预研），工业场景暂时保持运动学。
- [ ] **R4** 先做现有 ckpt 的真 L0（人提供 npz/LeRobot），不动物理。

默认若不选：下一实现轮次应按 **R1 → R2**，因为原方案写过「未标定禁止抓取成功率」，而现在最大的认知风险仍是把 GIF 当成抓取验证。

---

## 8. 本文件不覆盖的范围

- 众擎飞书 Native SDK wiki（需登录导出）
- HuggingFace GEAR-SONIC / BONES-SEED 权重与数据集是否已下载
- 实机现场网络拓扑以外的产线 WMS / L3 任务规划器
- 把 46 个 sibling PR 逐文件做质量审计（需要单独一轮，建议从 #47 diff 对 `wbc/` `runtime/` 抽样）
