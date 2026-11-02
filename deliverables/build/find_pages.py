import subprocess, json, re, sys
pdf = sys.argv[1]
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
txt = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"], capture_output=True, text=True).stdout for i in range(1, n + 1)]
keys = {"toc_abstract": r"^\s*ABSTRACT\s*$", "toc_problem": r"^\s*PROBLEM STATEMENT\s*$", "toc_method": r"^\s*METHODOLOGY / ARCHITECTURE\s*$",
        "toc_impl": r"^\s*IMPLEMENTATION . CODING AND OUTPUT SCREENSHOT\s*$", "toc_code": r"^\s*4\.2 Coding\s*$", "toc_shots": r"^\s*4\.3 Output Screenshot\s*$",
        "toc_conc": r"^\s*CONCLUSION\s*$", "toc_refs": r"^\s*REFERENCES\s*$"}
front = next(i for i, t in enumerate(txt) if re.search(keys["toc_abstract"], t, re.M))   # 0-based index of Abstract page
res = {}
for k, pat in keys.items():
    i = next(i for i, t in enumerate(txt) if i >= front and re.search(pat, t, re.M))
    res[k] = i - front + 1
print("front pages:", front, "total pages:", n, "body pages:", n - front)
print(res)
json.dump(res, open("pages.json", "w"))
