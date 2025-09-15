#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_cmd(py, script, fasta_dir, mmcif_dir, out_dir, aln_dir, config_preset, skip_relax):
    cmd = [
        py, str(script),
        str(fasta_dir),
        str(mmcif_dir),
        "--output_dir", str(out_dir),
        "--use_precomputed_alignments", str(aln_dir),
        "--config_preset", config_preset,
        "--model_device", "cuda:0",
    ]
    if skip_relax:
        cmd.append("--skip_relaxation")
    return cmd

def run_one(cmd, gpu_id, log_file=None, cwd=None):
    import io
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # 运行
    if log_file is None:
        # 捕获到内存里，方便做关键字判错
        p = subprocess.Popen(cmd, env=env, cwd=cwd,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        out, _ = p.communicate()
        ret = p.returncode
        text = out or ""
    else:
        # 写到文件里，结束后再读文件尾部判错
        with open(log_file, "w", buffering=1) as f:
            p = subprocess.Popen(cmd, env=env, cwd=cwd,
                                 stdout=f, stderr=subprocess.STDOUT, text=True)
            ret = p.wait()
        # 读取日志尾部若干 KB 做关键字判错
        try:
            with open(log_file, "r", errors="ignore") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                tail = 200_000  # 200KB 尾部足够
                f.seek(max(0, size - tail), os.SEEK_SET)
                text = f.read()
        except Exception:
            text = ""

    # 兜底关键字：即便 ret==0，但看到这些也判为失败
    err_markers = (
        "Traceback (most recent call last)",
        "ValueError:",
        "RuntimeError:",
        "AssertionError:",
        "Error:",
        "Exception:",
        "More than one input sequence found",
    )
    if ret == 0:
        lower = text  # 已经是 text
        if any(m in lower for m in err_markers):
            ret = 1

    return ret


def run_tasks(tasks, gpu_ids, script):
    """
    tasks: list[(cmd, gpu_id_placeholder, seed_name, log_file)]
    返回 (ok_seeds:list[str], fail_tasks:list[(cmd, gpu_id, seed_name, log_file)])
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ok, fail = [], []
    future_to_task = {}
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        for i, (cmd, _, seed_name, log_file) in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            fut = ex.submit(run_one, cmd, gpu_id, log_file, cwd=str(Path(cmd[1]).parent))
            future_to_task[fut] = (cmd, gpu_id, seed_name, log_file)

        for fut in as_completed(future_to_task):
            ret = fut.result()
            cmd, gpu_id, seed_name, log_file = future_to_task[fut]
            if ret == 0:
                ok.append(seed_name)
                print(f"[OK][GPU{gpu_id}][{seed_name}]")
            else:
                fail.append((cmd, gpu_id, seed_name, log_file))
                print(f"[FAIL][GPU{gpu_id}][{seed_name}] ret={ret}")

    return ok, fail


def main():
    ap = argparse.ArgumentParser(description="Batch run OpenFold across seed folders with GPU pinning and retries.")
    ap.add_argument("--seeds_root", required=True, help="根目录：包含多个 seed 子目录（如 seed_7778/ ...）")
    ap.add_argument("--output_root", required=True, help="输出根目录：每个 seed 的预测写到 <output_root>/<seed>/")
    ap.add_argument("--fasta_dir", required=True, help="FASTA 目录（run_pretrained_openfold.py 的第一个位置参数）")
    ap.add_argument("--mmcif_dir", required=True, help="mmCIF 目录（run_pretrained_openfold.py 的第二个位置参数）")
    ap.add_argument("--script", default="run_pretrained_openfold.py", help="脚本路径（默认：当前目录下 run_pretrained_openfold.py）")
    ap.add_argument("--config_preset", default="model_3_ptm", help="--config_preset，默认 model_3_ptm")
    ap.add_argument("--skip_relaxation", action="store_true", help="传入则加上 --skip_relaxation")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="GPU ID 列表，逗号分隔（默认 0-7）")
    ap.add_argument("--include", nargs="*", default=None, help="只运行指定 seeds（名称），例如 --include seed_7778 seed_8888")
    ap.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--log_dir", default=None, help="若提供则把每个 seed 的 stdout 写到 <log_dir>/<seed>.log")
    ap.add_argument("--max_retries", type=int, default=1, help="失败后最大重试次数（默认 1）")
    args = ap.parse_args()

    seeds_root = Path(args.seeds_root).resolve()
    output_root = Path(args.output_root).resolve()
    fasta_dir = Path(args.fasta_dir).resolve()
    mmcif_dir = Path(args.mmcif_dir).resolve()
    script = Path(args.script).resolve()

    if not seeds_root.is_dir():
        sys.exit(f"[ERROR] seeds_root 不存在或不是目录：{seeds_root}")
    if not fasta_dir.is_dir():
        sys.exit(f"[ERROR] fasta_dir 不存在或不是目录：{fasta_dir}")
    if not mmcif_dir.is_dir():
        sys.exit(f"[ERROR] mmcif_dir 不存在或不是目录：{mmcif_dir}")
    if not script.is_file():
        sys.exit(f"[ERROR] 脚本不存在：{script}")

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        sys.exit("[ERROR] 未提供有效 GPU ID 列表")

    seeds = [d for d in sorted(seeds_root.iterdir()) if d.is_dir()]
    if args.include:
        include_set = set(args.include)
        seeds = [d for d in seeds if d.name in include_set]
    if not seeds:
        sys.exit(f"[INFO] 在 {seeds_root} 下未发现 seed 子目录（或被 include 过滤为空）。")

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 计划运行 {len(seeds)} 个任务，使用 GPU: {gpu_ids}")

    # 构造任务
    tasks = []
    for i, seed_dir in enumerate(seeds):
        seed_name = seed_dir.name
        aln_dir = seed_dir  # 你的示例：--use_precomputed_alignments 指向 seed 目录
        out_dir = output_root / seed_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_cmd(
            py=sys.executable,
            script=script,
            fasta_dir=fasta_dir,
            mmcif_dir=mmcif_dir,
            out_dir=out_dir,
            aln_dir=aln_dir,
            config_preset=args.config_preset,
            skip_relax=args.skip_relaxation,
        )
        log_file = None if args.log_dir is None else str(Path(args.log_dir) / f"{seed_name}.log")
        tasks.append((cmd, -1, seed_name, log_file))  # gpu_id 此处占位，调度时分配

    # 打印命令
    for cmd, _, seed_name, _ in tasks:
        print("[CMD]", seed_name, " ".join(cmd))
    if args.dry_run:
        print("[DRY-RUN] 结束。")
        return

    # 第一次运行
    ok, fail = run_tasks(tasks, gpu_ids, script)
    print(f"[Round 1] 成功 {len(ok)}，失败 {len(fail)}")

    # 重试逻辑
    attempts = 0
    while fail and attempts < args.max_retries:
        attempts += 1
        print(f"[Retry #{attempts}] 重新运行 {len(fail)} 个失败任务...")
        ok2, fail2 = run_tasks(fail, gpu_ids, script)
        ok.extend(ok2)
        fail = fail2
        print(f"[Round {attempts+1}] 重试成功 {len(ok2)}，仍失败 {len(fail2)}")

    print(f"\n✅ 全部完成：成功 {len(ok)}，失败 {len(fail)}")
    if fail:
        print("仍失败的 seeds：", [x[2] for x in fail])

if __name__ == "__main__":
    main()
