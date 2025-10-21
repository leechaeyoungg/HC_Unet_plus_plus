# batch_eval_models.py
import os, re, csv, subprocess, sys

# === 설정 ===
ROOT = r"C:\Users\dromii\Downloads\crack_masks"
MODELS = [
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_90.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_80.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_70.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_60.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_50.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_40.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_30.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_20.pth",
    r"C:\Users\dromii\Downloads\815_hc_unetpp_final_epoch_10.pth",
]
EVAL_SCRIPT = r"eval_crack_iou.py" #모델 구조 정의
THR = 0.5
BATCH = 4
SIZE = 512

# === 실행 & 파싱 ===
def run_eval(weights_path):
    tag = os.path.splitext(os.path.basename(weights_path))[0]
    csv_out = f"{tag}_thr_{int(THR*1000):03d}.csv"
    log_out = f"{tag}_thr_{int(THR*1000):03d}.log"

    cmd = [sys.executable, EVAL_SCRIPT,
           "--root", ROOT,
           "--weights", weights_path,
           "--thr", str(THR),
           "--batch", str(BATCH),
           "--size", str(SIZE),
           "--csv", csv_out]

    print("\n>>>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    with open(log_out, "w", encoding="utf-8") as f:
        f.write(out)

    def grab(label):
        m = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)%", out)
        return float(m.group(1)) if m else None

    metrics = {
        "model": weights_path,
        "mIoU%": grab("mIoU (Crack/BG mean)"),
        "Crack IoU%": grab("Global Crack IoU"),
        "Background IoU%": grab("Global Background IoU"),
        "IoU on GT-positive images%": grab("IoU on GT-positive images"),
        "thr": THR,
        "csv": os.path.abspath(csv_out),
        "log": os.path.abspath(log_out),
    }
    print(f" -> mIoU={metrics['mIoU%']}%, Crack IoU={metrics['Crack IoU%']}%, BG IoU={metrics['Background IoU%']}%")
    return metrics

results = [run_eval(w) for w in MODELS]

# === 요약 출력 & 저장 ===
results.sort(key=lambda r: (r["mIoU%"] if r["mIoU%"] is not None else -1), reverse=True)
print("\n=== Summary (sorted by mIoU, thr=0.5) ===")
hdr = ["rank","mIoU%","Crack IoU%","Background IoU%","IoU on GT-positive images%","thr","model","csv","log"]
print("\t".join(hdr))
for i, r in enumerate(results, 1):
    row = [
        str(i),
        f"{r['mIoU%']:.2f}" if r["mIoU%"] is not None else "",
        f"{r['Crack IoU%']:.2f}" if r["Crack IoU%"] is not None else "",
        f"{r['Background IoU%']:.2f}" if r["Background IoU%"] is not None else "",
        f"{r['IoU on GT-positive images%']:.2f}" if r["IoU on GT-positive images%"] is not None else "",
        str(r["thr"]),
        r["model"],
        r["csv"],
        r["log"],
    ]
    print("\t".join(row))

with open("eval_summary_models.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(hdr[1:])
    for r in results:
        w.writerow([
            r["mIoU%"],
            r["Crack IoU%"],
            r["Background IoU%"],
            r["IoU on GT-positive images%"],
            r["thr"],
            r["model"],
            r["csv"],
            r["log"],
        ])
print("\nSaved: eval_summary_models.csv")
