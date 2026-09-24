import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.normalize import normalize_address, normalize_name, romanize_indic, skeleton  # noqa: E402


def test_native_script_meets_english():
    en = normalize_name("Royal Builders Private Limited")["name_skel"]
    hi = normalize_name("रॉयल बिल्डर्स प्राइवेट लिमिटेड")["name_skel"]
    assert en == hi == "ryl bldrs"


def test_bengali_and_constructions():
    assert normalize_name("স্মার্ট গুড এক্সপোর্টস প্রাইভেট লিমিটেড")["name_skel"] == \
        normalize_name("Smart Good Exports Private Limited")["name_skel"]
    assert skeleton(romanize_indic("कंस्ट्रक्शंस")) == skeleton("constructions")


def test_alias_web_and_noise():
    n = normalize_name("Rizacira formerly: Lopez, Smith and Campbell LLC")
    assert n["name_core"] == "lopez smith campbell" and n["name_has_alias"] and n["name_alias"] == "rizacira"
    assert normalize_name("literacycenter.com")["name_compact"] == "literacycenter"
    assert normalize_name("Literacy Center LLC")["name_compact"] == "literacycenter"
    assert normalize_name("regional prime horizon center (ID: 96415)")["name_core"] == "regional prime horizon center"
    assert normalize_name("LLC Orth0pedic Care")["name_core"] == "orthopedic care"
    assert normalize_name("Contre & Fils Distribution [S.A.R.L.]")["name_core"] == "contre fils distribution"


def test_address_canonical():
    a = normalize_address("0520 ELEVENTH STREET, BARTESVILLE, OK")
    b = normalize_address("520 11th Saint, Bartlesville, Oklahoma")
    assert a["addr_nums"] == b["addr_nums"] == "520 11"
    assert a["addr_clean"].split()[:3] == b["addr_clean"].split()[:3] == ["520", "11", "st"]
    assert b["addr_clean"].endswith("ok")
    f1 = normalize_address("63 R. DE DIEPPE, LILLE, Hauts-de-France")
    assert f1["addr_clean"].startswith("63 rue dieppe lille")
    assert normalize_address("")["addr_empty"]
