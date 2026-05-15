"""
main.py  –  SkipNet-CBAM Medical Imaging Experiment Entry Point
================================================================
两种使用方式：

【命令行】  python main.py --task both --suite
【PyCharm】 直接点运行 → 交互式菜单引导选择实验配置

命令行完整参数说明见底部 argparse 定义，或运行：
    python main.py --help
"""

from __future__ import annotations

import sys


# ─────────────────────────────────────────────────────────────────────────────
# 交互式菜单（PyCharm 直接运行时触发）
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, options: list[str], default: int = 0) -> int:
    """打印选项列表，返回用户选择的索引。"""
    print(prompt)
    for i, opt in enumerate(options):
        marker = "  [默认]" if i == default else ""
        print(f"  {i + 1}. {opt}{marker}")
    while True:
        raw = input(f"请输入序号 [1-{len(options)}，直接回车选默认]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  ⚠ 请输入 1 到 {len(options)} 之间的数字")


def _ask_int(prompt: str, default: int) -> int:
    """读取一个整数，直接回车使用默认值。"""
    raw = input(f"{prompt} [默认 {default}]: ").strip()
    if raw == "":
        return default
    if raw.isdigit():
        return int(raw)
    print(f"  ⚠ 输入无效，使用默认值 {default}")
    return default


def _separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def interactive_menu() -> list[str]:
    """
    引导用户完成实验配置，返回等价的 argv 列表供 argparse 解析。
    这样 PyCharm 和命令行走完全相同的代码路径，行为一致。
    """
    print()
    _separator("═")
    print("  SkipNet-CBAM 医学图像实验平台")
    print("  （直接回车 = 使用默认值）")
    _separator("═")
    print()

    argv: list[str] = []

    # ── 1. 实验类型 ────────────────────────────────────────────────────────
    _separator()
    task_idx = _ask(
        "① 选择实验任务：",
        [
            "查看数据集统计（不训练）",
            "仅 HEp-2 荧光图像分类",
            "仅 Camelyon 病理癌症检测",
            "两个数据集都跑  ★推荐",
        ],
        default=3,
    )
    task_map = {0: "summary", 1: "hep2", 2: "camelyon", 3: "both"}
    task = task_map[task_idx]
    argv += ["--task", task]

    if task == "summary":
        print()
        print("  → 将仅打印数据集统计信息，不进行任何训练。")
        return argv

    print()

    # ── 2. 运行模式 ────────────────────────────────────────────────────────
    _separator()
    mode_idx = _ask(
        "② 选择运行模式：",
        [
            "仅跑 Proposed（我们的算法）",
            "全套对比实验（4 个模型依次运行）  ★论文推荐",
        ],
        default=1,
    )
    if mode_idx == 1:
        argv += ["--suite"]
    else:
        # 单模型模式下让用户选具体模型
        print()
        _separator()
        model_idx = _ask(
            "   选择要运行的单个模型：",
            [
                "proposed       — SkipNet + CBAM（我们的算法）",
                "skipnet_no_cbam — SkipNet（无注意力机制消融）",
                "residual_cnn   — Residual CNN 基线",
                "simple_cnn     — Simple CNN 基线",
            ],
            default=0,
        )
        model_map = {0: "proposed", 1: "skipnet_no_cbam",
                     2: "residual_cnn", 3: "simple_cnn"}
        argv += ["--model", model_map[model_idx]]

    print()

    # ── 3. 数据量限制 ──────────────────────────────────────────────────────
    _separator()
    print("③ 数据量设置（控制运行时长）：")
    print()
    limit_idx = _ask(
        "   选择预设方案：",
        [
            "快速调试  — HEp-2: 2000 张 / Camelyon: 3000 张 / 5 epochs",
            "均衡实验  — HEp-2: 6000 张 / Camelyon: 8000 张（yaml 默认值）",
            "完整实验  — HEp-2: 12000 张 / Camelyon: 20000 张 / 30 epochs",
            "自定义    — 手动输入每个参数",
        ],
        default=1,
    )

    if limit_idx == 0:
        # 快速调试
        if task in ("hep2", "both"):
            argv += ["--hep2-max", "2000"]
        if task in ("camelyon", "both"):
            argv += ["--camelyon-max", "3000"]
        argv += ["--epochs", "5"]

    elif limit_idx == 1:
        # 使用 yaml 默认值，不追加任何参数
        pass

    elif limit_idx == 2:
        # 完整实验
        if task in ("hep2", "both"):
            argv += ["--hep2-max", "12000"]
        if task in ("camelyon", "both"):
            argv += ["--camelyon-max", "20000"]
        argv += ["--epochs", "30"]

    elif limit_idx == 3:
        # 自定义
        print()
        if task in ("hep2", "both"):
            n = _ask_int("   HEp-2 最大样本数（0 = 不限制）", default=6000)
            if n > 0:
                argv += ["--hep2-max", str(n)]
        if task in ("camelyon", "both"):
            n = _ask_int("   Camelyon 最大样本数（0 = 不限制）", default=8000)
            if n > 0:
                argv += ["--camelyon-max", str(n)]
        ep = _ask_int("   训练轮数 epochs（0 = 使用 yaml 默认值）", default=0)
        if ep > 0:
            argv += ["--epochs", str(ep)]

    print()

    # ── 4. 设备 ───────────────────────────────────────────────────────────
    _separator()
    dev_idx = _ask(
        "④ 计算设备：",
        [
            "auto  — 自动检测（有 GPU 用 GPU，否则 CPU）  ★推荐",
            "cuda  — 强制 GPU（cuda:0）",
            "cpu   — 强制 CPU",
        ],
        default=0,
    )
    dev_map = {0: "auto", 1: "cuda", 2: "cpu"}
    dev = dev_map[dev_idx]
    if dev != "auto":
        argv += ["--device", dev]

    print()

    # ── 5. 配置文件 ───────────────────────────────────────────────────────
    _separator()
    cfg_idx = _ask(
        "⑤ 配置文件：",
        [
            "使用默认配置 configs/default.yaml  ★推荐",
            "指定其他配置文件路径",
        ],
        default=0,
    )
    if cfg_idx == 1:
        cfg_path = input("   请输入配置文件路径: ").strip()
        if cfg_path:
            argv += ["--config", cfg_path]

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print()
    _separator("═")
    print("  实验配置确认")
    _separator()
    print(f"  等价命令：python main.py {' '.join(argv)}")
    _separator("═")
    print()
    confirm = input("确认开始实验？[Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("已取消，退出。")
        sys.exit(0)
    print()
    return argv


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from src.dip_medimg.cli import main as cli_main

    # 判断是否有命令行参数传入
    # sys.argv[0] 是脚本名，长度 > 1 说明用户传了实际参数
    if len(sys.argv) > 1:
        # ── 命令行模式：直接交给 cli_main 处理 ──────────────────────────
        cli_main()
    else:
        # ── PyCharm / 直接双击运行：启动交互式菜单 ──────────────────────
        try:
            argv = interactive_menu()
        except (KeyboardInterrupt, EOFError):
            print("\n\n已中断，退出。")
            sys.exit(0)

        # 将菜单结果注入 sys.argv 后调用同一个 cli_main
        sys.argv = [sys.argv[0]] + argv
        cli_main()


if __name__ == "__main__":
    main()
