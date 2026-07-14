import sys

BASE = r"C:\Users\ashis\AppData\Local\Temp\claude\C--work-java-OrderBook-rs\61d3e3c4-7df9-4bef-bb3c-d5cbdc4a99d6\scratchpad\crossport"

# labels produced by iterative fits -> looser 1e-6 relative tolerance
ITERATIVE = {"bs.iv.roundtrip", "bond.ytm", "zs.roundtrip", "pm.irr",
             "ns.b0", "ns.rmse", "ns.z5", "sv.b0", "sv.rmse", "sv.z5",
             "garch.omega", "garch.alpha", "garch.beta", "garch.fc1"}
TIGHT = 1e-9
LOOSE = 1e-6


def load(name):
    d = {}
    order = []
    with open(f"{BASE}\\{name}") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k] = float(v)
            order.append(k)
    return d, order


def main():
    a_name, b_name = sys.argv[1], sys.argv[2]
    a, order = load(a_name)
    b, _ = load(b_name)
    bad = 0
    for k in order:
        if k not in b:
            print(f"MISSING {k} in {b_name}")
            bad += 1
            continue
        x, y = a[k], b[k]
        tol = LOOSE if k in ITERATIVE else TIGHT
        denom = max(abs(x), abs(y), 1e-30)
        rel = abs(x - y) / denom
        absdiff = abs(x - y)
        if rel > tol and absdiff > 1e-12:
            print(f"MISMATCH {k}: {a_name}={x:.15e}  {b_name}={y:.15e}  rel={rel:.3e}")
            bad += 1
    extra = [k for k in b if k not in a]
    for k in extra:
        print(f"EXTRA {k} in {b_name}")
    print(f"--- {a_name} vs {b_name}: {len(order)} labels, {bad} mismatches ---")


if __name__ == "__main__":
    main()
