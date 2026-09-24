"""Per-file schema/missingness/country/script stats, streaming."""
import collections, re
DATA = "D:/6ab10eb3b23ba_student_resource/student_resource/dataset"
nonascii = re.compile(r"[^\x00-\x7f]")
indic = re.compile(r"[\u0900-\u0DFF]")
for split in ("train", "test"):
    for s in (1, 2, 3):
        path = f"{DATA}/{split}/{split}_source{s}.tsv"
        n = 0; bad = 0; cc = collections.Counter(); empty_name = empty_addr = 0
        name_indic = addr_indic = name_nonascii = 0; nl = al = 0; null_tok = 0
        cc_empty_addr = collections.Counter(); cc_indic_name = collections.Counter()
        with open(path, encoding="utf-8") as f:
            header = next(f).rstrip("\n").split("\t")
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) != 4: bad += 1; continue
                n += 1; _, name, addr, c = p
                cc[c] += 1
                if not name.strip(): empty_name += 1
                if not addr.strip(): empty_addr += 1; cc_empty_addr[c] += 1
                if indic.search(name): name_indic += 1; cc_indic_name[c] += 1
                if indic.search(addr): addr_indic += 1
                if nonascii.search(name): name_nonascii += 1
                if "NULL" in addr: null_tok += 1
                nl += len(name); al += len(addr)
        print(f"{split} S{s}: rows={n} badrows={bad} header={header}")
        print(f"   countries={dict(cc)}")
        print(f"   empty name={empty_name} empty addr={empty_addr} by country={dict(cc_empty_addr)} 'NULL' in addr={null_tok}")
        print(f"   indic-script name={name_indic} {dict(cc_indic_name)} indic addr={addr_indic} non-ascii name={name_nonascii}")
        print(f"   mean len name={nl/n:.1f} addr={al/n:.1f}")
