#!/usr/bin/env python3
"""Check every Obsidian wikilink in the given notes, plus table and math integrity.

Usage: python3 linkcheck.py NOTE.md [NOTE2.md ...]
Exit code 0 when everything resolves, 1 otherwise.
Checks: target file exists in the vault, `#Heading` exists as a heading line,
`#^block-id` exists as a block ID, tables keep a constant column count,
inline `$` and fenced code blocks are balanced.
"""
import os, re, sys

def vault_root(path):
    """The nearest folder above `path` holding .obsidian (an Obsidian vault), else the git repository root."""
    start = os.path.dirname(os.path.abspath(path))
    for marker in (".obsidian", ".git"):
        d = start
        while d != "/":
            if os.path.isdir(os.path.join(d, marker)):
                return d
            d = os.path.dirname(d)
    return start

def headings(path):
    hs = {re.sub(r"^#+\s*", "", l).strip() for l in open(path, encoding="utf-8", errors="ignore") if re.match(r"#+ ", l)}
    return hs | {re.sub(r"\s+", " ", h.replace(":", "")) for h in hs}  # Obsidian links drop colons

def strip_code(t):
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", t)

def main(files):
    V = vault_root(files[0])
    allf = [os.path.join(r, f) for r, _, fs in os.walk(V) if "/.trash" not in r and "/.obsidian" not in r for f in fs]
    def find(name, src):
        if "/" in name:
            for base in (os.path.dirname(src), V):
                for c in (os.path.join(base, name), os.path.join(base, name + ".md")):
                    if os.path.isfile(c): return c
        hits = [p for p in allf if p.endswith("/" + name) or p.endswith("/" + name + ".md")]
        return hits[0] if hits else None
    bad = 0
    for src in files:
        text = open(src, encoding="utf-8").read()
        n = b = 0
        for m in re.finditer(r"!?\[\[(.*?)\]\]", strip_code(text)):
            n += 1
            target = m.group(1).replace("\\|", "|").split("|")[0]
            name, _, h = target.partition("#")
            tgt = src if name == "" else find(name, src)
            if tgt is None:
                print(f"MISSING FILE: {name}"); b += 1; continue
            if not h or not tgt.endswith(".md"): continue
            if h.startswith("^"):
                if not re.search(r"(^|\s)" + re.escape(h) + r"\s*$", open(tgt, encoding="utf-8").read(), re.M):
                    print(f"MISSING BLOCK: {name or '(self)'}#{h}"); b += 1
            elif h.strip() not in headings(tgt):
                print(f"MISSING HEADING: {name or '(self)'}#{h}"); b += 1
        # table, math and fence integrity
        lines = text.split("\n"); fence = math = False; cols = None
        for i, l in enumerate(lines, 1):
            if re.sub(r"^(>\s?)+", "", l).strip().startswith("```"): fence = not fence; continue  # fences inside callouts too
            if fence: continue
            if l.strip() == "$$": math = not math; continue
            if math: continue
            t = re.sub(r"`[^`]*`", "", l).replace("\\$", "")
            if t.count("$") % 2: print(f"ODD $ line {i}"); b += 1
            if l.startswith("|"):
                c = re.sub(r"\[\[[^\]]*\]\]", "L", re.sub(r"\$[^$]*\$", "M", t)).replace("\\|", "").count("|")
                if cols is None: cols = c
                elif c != cols: print(f"TABLE ROW line {i}: {c} pipes, expected {cols}"); b += 1
            else:
                cols = None
        if fence or math: print("UNCLOSED code fence or $$ block"); b += 1
        print(f"{os.path.basename(src)}: {n} links, {b} problems"); bad += b
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
