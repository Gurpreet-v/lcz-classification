# Run:  streamlit run streamlit_app.py
import streamlit as st
from typing import Dict, Any, Tuple, List, Optional
import json
import os
import pandas as pd
import io
from PIL import Image

st.set_page_config(page_title="LCZ Classifier", page_icon="", layout="wide")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CUR_DIR)

def parse_bounds(spec: str):
    if spec is None:
        return None
    s = str(spec).strip()
    if s == "" or s == "-" or s.lower() in {"na", "n/a", "none"}:
        return (None, None, "na")
    s = s.replace("–", "-").replace("—", "-").replace("≥", ">=").replace("≤", "<=")
    if s.startswith(">="):
        return (float(s[2:]), None, "ge")
    if s.startswith("<="):
        return (None, float(s[2:]), "le")
    if s.startswith(">"):
        return (float(s[1:]), None, "gt")
    if s.startswith("<"):
        return (None, float(s[1:]), "lt")
    if "-" in s:
        parts = [p.strip() for p in s.split("-")]
        if len(parts) == 2 and parts[0] and parts[1]:
            try:
                return (float(parts[0]), float(parts[1]), "range")
            except:
                pass
    try:
        val = float(s)
        return (val, val, "eq")
    except:
        return None

def value_in_spec(value: Optional[float], spec: str) -> Optional[bool]:
    pr = parse_bounds(spec)
    if pr is None:
        return None
    lo, hi, kind = pr
    if value is None:
        return False if kind != "na" else True
    if kind == "na":
        return True
    if kind == "range":
        return (lo is None or value >= lo) and (hi is None or value <= hi)
    if kind == "gt":
        return value > (lo if lo is not None else float("-inf"))
    if kind == "lt":
        return value < (hi if hi is not None else float("inf"))
    if kind == "ge":
        return value >= (lo if lo is not None else float("-inf"))
    if kind == "le":
        return value <= (hi if hi is not None else float("inf"))
    if kind == "eq":
        return abs(value - (lo if lo is not None else value)) < 1e-9
    return None

def range_distance(value: Optional[float], spec: str) -> float:
    pr = parse_bounds(spec)
    if pr is None:
        return 0.0
    lo, hi, kind = pr
    if value is None:
        return 0.5
    if kind == "na":
        return 0.0
    if kind in {"gt","ge"}:
        bound = lo if lo is not None else float("-inf")
        return max(0.0, bound - value)
    if kind in {"lt","le"}:
        bound = hi if hi is not None else float("inf")
        return max(0.0, value - bound)
    if kind == "eq":
        target = lo if lo is not None else value
        return abs(value - target)
    if kind == "range":
        lo_b = lo if lo is not None else float("-inf")
        hi_b = hi if hi is not None else float("inf")
        if lo_b <= value <= hi_b:
            return 0.0
        if value < lo_b:
            return lo_b - value
        return value - hi_b
    return 0.0

def to_num(val):
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except:
        return None

DEFAULTS = {
    "SVF": 0.85, "SCR": 0.20, "FAR": 0.90,
    "BSF": 25.0, "ISF": 45.0, "PSF": 15.0,
    "BH": 8.0, "BHD": 8.0, "BHV": 120.0,
    "AL": 0.20, "TR": 5.0, "TH": "",
}

def init_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _epsilon_for(param: str) -> float:
    eps = {"SVF":0.01,"SCR":0.05,"FAR":0.05,"AL":0.01,
           "BSF":1.0,"ISF":1.0,"PSF":1.0,"BH":0.5,"BHD":0.5,"BHV":5.0,"TR":0.5}
    return eps.get(param, 0.1)

def suggest_value_from_spec(param: str, spec: str):
    pr = parse_bounds(spec)
    if pr is None: return None
    lo, hi, kind = pr
    eps = _epsilon_for(param)
    if kind == "na": return None
    if kind == "eq": return lo
    if kind == "range":
        if lo is None and hi is not None: return max(0.0, hi - eps)
        if hi is None and lo is not None: return lo + eps
        return (lo + hi) / 2.0
    if kind in {"ge","gt"}: return (lo if lo is not None else 0.0) + eps
    if kind in {"le","lt"}: return max(0.0, (hi if hi is not None else 0.0) - eps)
    return lo


PRESET_OVERRIDES = {
    
    "1":  {"BSF": 41.0, "BH": 26.0},         
    "2":  {"BSF": 40.0, "BH": 20.0, "BHD": 8.5},   
    "34": {"BSF": 45.0, "BH": 20.0, "BHD": 10.0, "BHV": 160.0}, 
    "35": {"BSF": 45.0, "BH": 20.0, "BHD": 10.0, "BHV": 140.0}, 
    "4":  {"BSF": 30.0, "BH": 26.0},
    "5":  {"BSF": 30.0, "BH": 20.0},
    "6":  {"BSF": 30.0, "BH": 8.5},
    "8":  {"BSF": 25.0, "BH": 8.5, "PSF": 5.0},      
    "8B": {"BSF": 25.0, "BH": 8.5, "PSF": 15.0},     
    "9":  {"BSF": 5.0,  "SVF": 0.85},                
    "A":  {"BSF": 9.0,  "ISF": 9.0, "PSF": 95.0},   
    "B":  {"BSF": 9.0,  "ISF": 9.0, "PSF": 92.0},
    "C":  {"BSF": 9.0,  "ISF": 9.0, "PSF": 92.0, "TH": 'bush'},
    "D":  {"BSF": 9.0,  "ISF": 9.0, "PSF": 92.0},
    "G":  {"BSF": 0.0,  "ISF": 0.0, "AL": 0.06, "TR": 1.0},
}

def fill_preset_from_lcz(lcz_code: str):
    spec = LCZ_TABLE.get(lcz_code, {})
    
    for p in ["SVF","SCR","FAR","BSF","ISF","PSF","BH","BHD","BHV","AL","TR"]:
        if p in spec:
            val = suggest_value_from_spec(p, spec[p])
            if val is not None:
                st.session_state[p] = float(val)
    if "TH" in spec and spec["TH"] not in ("-", "", None):
        st.session_state["TH"] = spec["TH"]

    if lcz_code in PRESET_OVERRIDES:
        for k, v in PRESET_OVERRIDES[lcz_code].items():
            st.session_state[k] = v

def apply_preset_to_session(lcz_code: str):
    spec = LCZ_TABLE.get(lcz_code, {})
    
    for p in ["SVF","SCR","FAR","BSF","ISF","PSF","BH","BHD","BHV","AL","TR"]:
        if p in spec:
            v = suggest_value_from_spec(p, spec[p])
            if v is not None:
                st.session_state[p] = float(v)
    
    if "TH" in spec and spec["TH"] not in ("-","",None):
        st.session_state["TH"] = spec["TH"]
    
    for k, v in PRESET_OVERRIDES.get(lcz_code, {}).items():
        st.session_state[k] = v


LCZ_TABLE: Dict[str, Dict[str, str]] = {
    "1":  {"SVF":"0.2-0.4","SCR":">2","FAR":">=3","BSF":"30-45","ISF":"40-60","PSF":"<10","BH":">=24","BHD":"-","BHV":"-","AL":"0.1-0.2","TH":"-","TR":"8"},
    "2":  {"SVF":"0.3-0.6","SCR":"0.75-2","FAR":"1.5-3","BSF":"30-45","ISF":"30-50","PSF":"<20","BH":"9-24","BHD":"<9","BHV":"-","AL":"0.1-0.2","TH":"-","TR":"6-7"},
    "3":  {"SVF":"0.2-0.6","SCR":"0.75-1.5","FAR":"1-1.5","BSF":"40-70","ISF":"20-50","PSF":"<30","BH":"<=9","BHD":"-","BHV":"-","AL":"0.1-0.2","TH":"-","TR":"6"},
    "34": {"SVF":"0.2-0.4","SCR":">2","FAR":"1.2-2.5","BSF":"25-30","ISF":"40-60","PSF":"<10","BH":"12-40","BHD":"-","BHV":">145","AL":"0.12-0.25","TH":"-","TR":"8"},
    "4":  {"SVF":"0.5-0.7","SCR":"0.75-1.25","FAR":"2-3","BSF":"20-30","ISF":"30-40","PSF":"30-40","BH":">=24","BHD":"-","BHV":"-","AL":"0.12-0.25","TH":"-","TR":"7-8"},
    "35": {"SVF":"0.3-0.6","SCR":"0.75-2","FAR":"1.2-2","BSF":"20-30","ISF":"30-50","PSF":"<20","BH":"12-20","BHD":"-","BHV":"<=145","AL":"0.1-0.2","TH":"-","TR":"6-7"},
    "5":  {"SVF":"0.5-0.8","SCR":"0.3-0.75","FAR":"1.2-2","BSF":"20-30","ISF":"30-50","PSF":"20-40","BH":"9-24","BHD":"-","BHV":"-","AL":"0.12-0.25","TH":"-","TR":"5-6"},
    "6":  {"SVF":"0.6-0.9","SCR":"0.3-0.75","FAR":"<0.3","BSF":"20-40","ISF":"20-50","PSF":"30-60","BH":"<=9","BHD":"-","BHV":"-","AL":"0.12-0.25","TH":"-","TR":"5"},
    "8":  {"SVF":">0.7","SCR":"0.1-0.3","FAR":"0.8-2.5","BSF":"30-50","ISF":"40-50","PSF":"<20","BH":"<=9","BHD":"-","BHV":"-","AL":"0.15-0.25","TH":"-","TR":"5"},
    "8B": {"SVF":">0.7","SCR":"0.1-0.3","FAR":"0.8-2.5","BSF":"30-50","ISF":"40-50","PSF":"<20","BH":"<=9","BHD":"-","BHV":"-","AL":"0.12-0.25","TH":"-","TR":"5"},
    "9":  {"SVF":">0.8","SCR":"0.1-0.25","FAR":"<=0.3","BSF":"<10","ISF":"<20","PSF":"60-80","BH":"<=9","BHD":"-","BHV":"-","AL":"0.12-0.25","TH":"-","TR":"5-6"},
    "A":  {"SVF":"<0.4","SCR":">1","FAR":"-","BSF":"<10","ISF":"<10","PSF":">90","BH":"3-30","BHD":"-","BHV":"-","AL":"0.1-0.2","TH":"Dense trees","TR":"8"},
    "B":  {"SVF":"0.5-0.8","SCR":"0.25-0.75","FAR":"-","BSF":"<10","ISF":"<10","PSF":">90","BH":"3-15","BHD":"-","BHV":"-","AL":"0.15-0.25","TH":"Sparse trees","TR":"5-6"},
    "C":  {"SVF":"0.7-0.9","SCR":"0.25-1.0","FAR":"-","BSF":"<10","ISF":"<10","PSF":">90","BH":"<2","BHD":"-","BHV":"-","AL":"0.15-0.3","TH":"Dense bush","TR":"4-5"},
    "D":  {"SVF":">0.9","SCR":"<0.1","FAR":"-","BSF":"<10","ISF":"<10","PSF":">90","BH":"<1","BHD":"-","BHV":"-","AL":"0.15-0.25","TH":"Ground cover area","TR":"3-4"},
    "G":  {"SVF":">0.9","SCR":"<0.1","FAR":"-","BSF":"<10","ISF":"<10","PSF":">90","BH":"-","BHD":"-","BHV":"-","AL":"0.02-0.10","TH":"-","TR":"1"},
}

LCZ_DEFINITIONS = {
    "1": {
        "type": "Compact high-rise",
        "definition": "Dense mix of tall buildings (tens of stories). Few or no trees. Land cover mostly paved. Materials: concrete, steel, stone, and glass."
    },
    "2": {
        "type": "Compact midrise",
        "definition": "Dense mix of midrise buildings (3–9 stories). Few or no trees. Land cover mostly paved. Materials: stone, brick, tile, concrete."
    },
    "3": {
        "type": "Compact low-rise",
        "definition": "Dense mix of low-rise buildings (1–3 stories). Few or no trees. Land cover mostly paved. Materials: stone, brick, tile, concrete."
    },
    "34": {
        "type": "Mixed-use high-rise",
        "definition": "Mixed-use built-up zone combining compact high-rise buildings with low or midrise features"
    },
    "4": {
        "type": "Open high-rise",
        "definition": "Open arrangement of tall buildings (tens of stories). Abundance of pervious land cover (low plants, scattered trees). Materials: concrete, steel, stone, glass."
    },
    "35": {
        "type": "Mixed-use midrise",
        "definition": "Mixed-use built-up zone combining compact midrise buildings with large low-rise elements"
    },
    "5": {
        "type": "Open midrise",
        "definition": "Open arrangement of midrise buildings (3–9 stories). Abundance of pervious land cover (low plants, scattered trees). Materials: concrete, steel, stone, glass."
    },
    "6": {
        "type": "Open low-rise",
        "definition": "Open arrangement of low-rise buildings (1–3 stories). Abundance of pervious land cover (low plants, scattered trees). Materials: wood, brick, stone, tile, concrete."
    },
    "8": {
        "type": "Large low-rise",
        "definition": "Open arrangement of large low-rise buildings (1–3 stories). Few or no trees. Land cover mostly paved. Materials: steel, concrete, metal, stone."
    },
    "8B": {
        "type": "Heavy industry",
        "definition": "Open arrangement of low-rise structures. Presence of street trees and lawns. Land cover mostly paved. Materials: metal, steel, concrete."
    },
    "9": {
        "type": "Sparsely built",
        "definition": "Sparse arrangement of small or medium-sized buildings in a natural setting. Abundance of pervious land cover (low plants, scattered trees)."
    },
    "A": {
        "type": "Dense trees",
        "definition": "Heavily wooded landscape of deciduous and/or evergreen trees. Land cover mostly pervious with low plants. Functions as natural forest, tree cultivation, or urban park."
    },
    "B": {
        "type": "Scattered trees",
        "definition": "Lightly wooded landscape of deciduous and/or evergreen trees. Land cover mostly pervious with low plants. Functions as natural forest, tree cultivation, or urban park."
    },
    "C": {
        "type": "Bush, scrub",
        "definition": "Open arrangement of bushes, shrubs, and short woody trees. Land cover mostly pervious (bare soil or sand). Functions as natural scrubland or agriculture."
    },
    "D": {
        "type": "Low plants",
        "definition": "Featureless landscape of grass or herbaceous plants/crops. Few or no trees. Zone functions as natural grassland, agriculture, or urban park."
    },
    "G": {
        "type": "Water",
        "definition": "Large open water bodies (seas, lakes) or smaller water bodies (rivers, reservoirs, lagoons)."
    }
}



# LCZ summary dictionary
LCZ_SUMMARY = [
    {"LCZ":"1", "Type":"Compact High-Rise",
     "Meteorology":"Strong UHI, low SVF, poor ventilation, trapped pollutants.",
     "Social":"Dense population, high cooling demand, heat stress risk."},
    
    {"LCZ":"2", "Type":"Compact Mid-Rise",
     "Meteorology":"Hot, moderate ventilation, canyon effect increases air pollution.",
     "Social":"Affordable housing. Great potential for Bodegas"},
    
    {"LCZ":"3", "Type":"Compact Low-Rise",
     "Meteorology":"Moderate heat retention, roofs exposed → faster night cooling.",
     "Social":"Often informal/low-income, vulnerable to heat & flooding. Great potential for Bodegas"},
    
    {"LCZ":"34", "Type":"Mixed-Use High-Rise",
     "Meteorology":"Heterogeneous microclimate, turbulence, localized hotspots.",
     "Social":"Commercial + residential mix → pollution & noise exposure."},
    
    {"LCZ":"35", "Type":"Mixed-Use Mid-Rise",
     "Meteorology":"Similar to LCZ 34, but with midrise dominance.",
     "Social":"Mixed comfort, uneven exposure."},
    
    {"LCZ":"4", "Type":"Open High-Rise",
     "Meteorology":"Higher SVF → better night cooling, still strong UHI.",
     "Social":"Planned luxury towers, greener courtyards, more resilience."},
    
    {"LCZ":"5", "Type":"Open Mid-Rise",
     "Meteorology":"Balanced canyons, moderate UHI, decent ventilation.",
     "Social":"Middle-class housing, better comfort than compact cores."},
    
    {"LCZ":"6", "Type":"Open Low-Rise",
     "Meteorology":"Cooler than compact forms, more pervious surfaces.",
     "Social":"Suburban housing, higher commuting-related energy demand."},
    
    {"LCZ":"8", "Type":"Large Low-Rise",
     "Meteorology":"Warehouses create daytime hotspots, slow night cooling.",
     "Social":"Industrial/commercial zones, workplace health risks."},
    
    {"LCZ":"8B", "Type":"Heavy Industry",
     "Meteorology":"Extra heat & emissions, strong nocturnal UHI.",
     "Social":"Environmental risks for like communities."},
    
    {"LCZ":"9", "Type":"Sparsely Built",
     "Meteorology":"Cooler due to vegetation, patchy heating depending on surfaces.",
     "Social":"Peri-urban/rural edges, liveable but under expansion pressure."},
    
    {"LCZ":"A", "Type":"Dense Trees",
     "Meteorology":"Strong evapotranspiration cooling, high humidity, carbon sink.",
     "Social":"Forests/parks, recreation, mental health benefits."},
    
    {"LCZ":"B", "Type":"Scattered Trees",
     "Meteorology":"Partial shading, fragmented cooling effect.",
     "Social":"Suburban parks, street trees improve liveability."},
    
    {"LCZ":"C", "Type":"Bush/Scrub",
     "Meteorology":"Limited evapotranspiration, reflects more radiation.",
     "Social":"Semi-arid/agricultural zones, limited cooling role."},
    
    {"LCZ":"D", "Type":"Low Plants/Grass",
     "Meteorology":"Evapotranspiration cooling, weaker than trees.",
     "Social":"Urban parks, agriculture, sports fields; social/health benefits."},
    
    {"LCZ":"G", "Type":"Water",
     "Meteorology":"Thermal regulator: cooler in day, warmer at night; increases humidit.",
     "Social":"Water supply, recreation, but flood risks if unmanaged."}
]

df_summary = pd.DataFrame(LCZ_SUMMARY)


LCZ_IMAGES = {
    "1": os.path.join(CUR_DIR, "images", "lcz1.png"),
    "2": os.path.join(CUR_DIR, "images", "lcz2.png"),
    "3": os.path.join(CUR_DIR, "images", "lcz3.png"),
    "34": os.path.join(CUR_DIR, "images", "lcz34.png"),
    "4": os.path.join(CUR_DIR, "images", "lcz4.png"),
    "35": os.path.join(CUR_DIR, "images", "lcz35.png"),
    "5": os.path.join(CUR_DIR, "images", "lcz5.png"),
    "6": os.path.join(CUR_DIR, "images", "lcz6.png"),
    "8": os.path.join(CUR_DIR, "images", "lcz8.png"),
    "8B": os.path.join(CUR_DIR, "images", "lcz8B.png"),
    "9": os.path.join(CUR_DIR, "images", "lcz9.png"),

    "A": os.path.join(CUR_DIR, "images", "lcza.png"),
    "B": os.path.join(CUR_DIR, "images", "lczb.png"),
    "C": os.path.join(CUR_DIR, "images", "lczc.png"),
    "D": os.path.join(CUR_DIR, "images", "lczd.png"),
    "G": os.path.join(CUR_DIR, "images", "lczg.png"),
}




def score_class(params: Dict[str, Any], lcz: str):
    spec = LCZ_TABLE[lcz]
    hits = 0
    dist = 0.0
    for k, spec_str in spec.items():
        if k == "TH":
            th = params.get("TH")
            if th and str(spec_str).lower() in str(th).lower():
                hits += 1
            continue
        val = to_num(params.get(k))
        inside = value_in_spec(val, spec_str)
        if inside is True: hits += 1
        dist += range_distance(val, spec_str)
    return hits, dist

def best_matches(params: Dict[str, Any], candidates=None, topk=3):
    if candidates is None:
        candidates = list(LCZ_TABLE.keys())
    scored = []
    for l in candidates:
        h, d = score_class(params, l)
        scored.append((l, h, d))
    scored.sort(key=lambda x: (-x[1], x[2], x[0]))
    return scored[:topk]



TH_TO_LCZ = {
    
    "dense tree": "A",          
    "dense": "A",          
    "scattered tree": "B",      
    "sparse tree": "B",
    "scattered": "B",      
    "sparse": "B",
    "bush": "C",                
    "scrub": "C",
    "low plant": "D",           
    "grass": "D",
    "ground cover": "D",
    "water": "G",               
    "lake": "G",
    "river": "G",
}


TR_BANDS = [
    ("A", (7.5, 8.5), "TR ≈ 8 → (dense trees)"),
    ("B", (4.5, 6.5), "TR ≈ 5–6 → (scattered/sparse trees)"),
    ("C", (3.5, 5.0), "TR ≈ 4–5 → (bush/scrub)"),
    ("D", (2.5, 4.0), "TR ≈ 3–4 → (low plants/grass)"),
    ("G", (0.5, 1.5), "TR ≈ 1 → (water)"),
]

def infer_lcz_from_TH(TH: str) -> tuple[str, str] | tuple[None, str]:
    if not TH:
        return None, "TH not provided"
    s = TH.lower()
    for key, code in TH_TO_LCZ.items():
        if key in s:
            return code, f"TH = '{TH}' → LCZ-{code}"
    return None, f"TH provided ('{TH}') but no keyword match"

def infer_lcz_from_TR(TR: float | None) -> tuple[str, str] | tuple[None, str]:
    if TR is None:
        return None, "TR not provided"
    for code, (lo, hi), label in TR_BANDS:
        if lo <= TR <= hi:
            return code, f"TR = {int(TR)} in {label} → LCZ-{code}"
    # nearest band fallback
    nearest = min(TR_BANDS, key=lambda b: min(abs(TR-b[1][0]), abs(TR-b[1][1])))
    code, (lo, hi), label = nearest
    return code, f"TR = {int(TR)} nearest to {label} → LCZ-{code}"

def prefer_water_with_AL(code: str | None, AL: float | None, TR: float | None) -> tuple[str | None, str | None]:
    if AL is not None and 0.02 <= AL <= 0.10:
        if TR is None or (0.5 <= TR <= 1.5):
            return "G", f"AL={AL} (0.02–0.10) and TR≈1 ⇒ LCZ-G (water)"
    return code, None




def classify_lcz(params: dict) -> dict:
    res = classify_flow(params)
    lcz = res["lcz"]

    res["type"] = LCZ_DEFINITIONS.get(lcz, {}).get("type", "Unknown")
    res["definition"] = LCZ_DEFINITIONS.get(lcz, {}).get("definition", "")
    res["image"] = LCZ_IMAGES.get(lcz, None)

    return res


def classify_flow(params: Dict[str, Any]):
    BSF = to_num(params.get("BSF"))
    ISF = to_num(params.get("ISF"))
    SVF = to_num(params.get("SVF"))
    BH  = to_num(params.get("BH"))
    BHD = to_num(params.get("BHD"))
    BHV = to_num(params.get("BHV"))
    SCR = to_num(params.get("SCR"))
    FAR = to_num(params.get("FAR"))
    PSF = to_num(params.get("PSF"))

    trace: List[str] = []
    nodes = set()
    edges = []

    def mark(frm, to, label=""):
        nodes.add(frm); nodes.add(to); edges.append((frm, to, label))

    nodes.add("start")
    built = (BSF is not None and BSF >= 10) or (ISF is not None and ISF >= 10)
    if built:
        trace.append("Built-up (BSF ≥10 or ISF ≥10)")
        mark("start", "built", "BSF≥10 or ISF≥10")

        if (BSF is not None and BSF < 20) and (SVF is not None and SVF > 0.8):
            trace.append("BSF < 20 and SVF > 0.8 ⇒ LCZ-9")
            mark("built", "lcz9", "BSF<20 & SVF>0.8")
            return {"lcz":"9","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}

        if BSF is not None and BSF >= 40:
            trace.append("BSF ≥ 40 → {1,2,3,34,35}")
            mark("built", "bsf40", "BSF≥40")
            if BH is not None:
                if BH >= 25:
                    trace.append("BH ≥ 25 ⇒ LCZ-1")
                    mark("bsf40", "lcz1", "BH≥25")
                    return {"lcz":"1","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}
                elif 10 <= BH < 25:

                    trace.append("Decide among LCZ-2 / LCZ-34 / LCZ-35 using BHD and BHV")
                    mark("bsf40", "midrise123435", "10≤BH<25")

                    if BHD is not None and BHD < 9:
                        trace.append(f"BHD={BHD} < 9 ⇒ LCZ-2")
                        mark("midrise123435", "lcz2", "BHD<9")
                        return {"lcz": "2", "trace": trace, "nodes": list(nodes), "edges": edges,
                                "alternatives": best_matches(params, ["34","35"], topk=2)}

                    if BHV is not None and BHV > 145:
                        trace.append(f"BHV={BHV} > 145 ⇒ LCZ-34")
                        mark("midrise123435", "lcz34", "BHV>145")
                        return {"lcz": "34", "trace": trace, "nodes": list(nodes), "edges": edges,
                                "alternatives": best_matches(params, ["2","35"], topk=2)}

                    if BHV is not None:
                        trace.append(f"BHV={BHV} ≤ 145 ⇒ LCZ-35")
                        mark("midrise123435", "lcz35", "BHV≤145")
                        return {"lcz": "35", "trace": trace, "nodes": list(nodes), "edges": edges,
                                "alternatives": best_matches(params, ["2","34"], topk=2)}

                    trace.append("BHD/BHV missing ⇒ fallback table scoring among {2,34,35}")
                    winner = best_matches(params, ["2","34","35"], topk=3)
                    mark("midrise123435", f"lcz{winner[0][0]}", "table-fallback")
                    return {"lcz": winner[0][0], "trace": trace, "nodes": list(nodes), "edges": edges,
                            "alternatives": winner}
                else:
                    trace.append("BH < 10 ⇒ LCZ-3")
                    mark("bsf40", "lcz3", "BH<10")
                    return {"lcz":"3","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}
            else:
                winner = best_matches(params, ["1","2","3","34","35"], topk=3)
                mark("bsf40", f"lcz{winner[0][0]}", "best-match (BH missing)")
                return {"lcz":winner[0][0],"trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":winner}
        else:
            trace.append("20% ≤ BSF < 40 (or missing high) → {4,5,6,8,8B}")
            mark("built", "bsf20to40", "20≤BSF<40")
            if BH is not None:
                if BH >= 25:
                    trace.append("BH ≥ 25 ⇒ LCZ-4")
                    mark("bsf20to40", "lcz4", "BH≥25")
                    return {"lcz":"4","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}
                elif 10 <= BH < 25:
                    trace.append("10 ≤ BH < 25 ⇒ LCZ-5")
                    mark("bsf20to40", "lcz5", "10≤BH<25")
                    return {"lcz":"5","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}
                else:
                    trace.append("BH < 10 ⇒ decide with SCR & FAR")
                    mark("bsf20to40", "lowrise_branch", "BH<10")
                    if SCR is not None and FAR is not None:
                        if (0.3 <= SCR <= 0.75 and FAR <= 0.3):
                            trace.append("0.3 ≤ SCR ≤ 0.75 and FAR ≤ 0.3 ⇒ LCZ-6")
                            mark("lowrise_branch", "lcz6", "SCR/FAR rule")
                            return {"lcz":"6","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}
                        elif (0.1 <= SCR < 0.3 and FAR > 0.3):
                            if params.get("PSF") is not None and float(params["PSF"]) < 10:
                                trace.append("PSF < 10 ⇒ LCZ-8")
                                mark("lowrise_branch", "lcz8", "PSF<10")
                                return {"lcz":"8","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}
                            else:
                                trace.append("PSF ≥ 10 ⇒ LCZ-8B")
                                mark("lowrise_branch", "lcz8B", "PSF≥10")
                                return {"lcz":"8B","trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":best_matches(params)}
                        else:
                            winner = best_matches(params, ["6","8","8B"], topk=3)
                            mark("lowrise_branch", f"lcz{winner[0][0]}", "best-match")
                            return {"lcz":winner[0][0],"trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":winner}
                    else:
                        winner = best_matches(params, ["6","8","8B"], topk=3)
                        mark("lowrise_branch", f"lcz{winner[0][0]}", "best-match (SCR/FAR missing)")
                        return {"lcz":winner[0][0],"trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":winner}
            else:
                winner = best_matches(params, ["4","5","6","8","8B"], topk=3)
                mark("bsf20to40", f"lcz{winner[0][0]}", "best-match (BH missing)")
                return {"lcz":winner[0][0],"trace":trace,"nodes":list(nodes), "edges":edges, "alternatives":winner}
            
    else:
        trace.append("Land-cover branch (BSF <10 and ISF <10)")
        mark("start", "land", "BSF<10 & ISF<10")


        TR = to_num(params.get("TR"))
        AL = to_num(params.get("AL"))
        TH = params.get("TH")


        code, reason = infer_lcz_from_TH(TH)
        if code is not None:

            code2, water_reason = prefer_water_with_AL(code, AL, TR)
            if code2 != code:
                code = code2
                reason = water_reason
            trace.append(reason)
            mark("land", f"lcz{code}", "TH rule")
            return {"lcz": code, "trace": trace, "nodes": list(nodes), "edges": edges,
                    "alternatives": best_matches(params, ["A","B","C","D","G"], topk=3)}


        code, reason = infer_lcz_from_TR(TR)
        if code is not None:
            code2, water_reason = prefer_water_with_AL(code, AL, TR)
            if code2 != code:
                code = code2
                reason = water_reason
            trace.append(reason)
            mark("land", f"lcz{code}", "TR rule")
            return {"lcz": code, "trace": trace, "nodes": list(nodes), "edges": edges,
                    "alternatives": best_matches(params, ["A","B","C","D","G"], topk=3)}


        if AL is not None and 0.02 <= AL <= 0.10:
            trace.append(f"AL={AL} within 0.02–0.10 (water-like) ⇒ LCZ-G")
            mark("land", "lczG", "AL rule")
            return {"lcz": "G", "trace": trace, "nodes": list(nodes), "edges": edges,
                    "alternatives": best_matches(params, ["A","B","C","D"], topk=3)}


        candidates = ["A","B","C","D","G"]
        winner = best_matches(params, candidates, topk=3)
        trace.append("Insufficient TH/TR/AL — fell back to land-cover table scoring.")
        mark("land", f"lcz{winner[0][0]}", "table fallback")
        return {"lcz": winner[0][0], "trace": trace, "nodes": list(nodes), "edges": edges,
                "alternatives": winner}


def to_dot(nodes: List[str], edges: List[Tuple[str,str,str]], final_label: str):

    label_map = {
        "start":"LCZ parameters",
        "built":"Built-up branch",
        "lcz9":"LCZ-9",
        "bsf40":"BSF ≥ 40%",
        "midrise123435":"10 ≤ BH < 25 → {2,34,35}",
        "lcz1":"LCZ-1",
        "lcz3":"LCZ-3",
        "bsf20to40":"20% ≤ BSF < 40%",
        "lcz4":"LCZ-4",
        "lcz5":"LCZ-5",
        "lowrise_branch":"BH < 10 → (SCR/FAR)",
        "lcz6":"LCZ-6",
        "lcz8":"LCZ-8",
        "lcz8B":"LCZ-8B",
        "land":"Land-cover branch",
        "lczA":"LCZ-A",
        "lczB":"LCZ-B",
        "lczC":"LCZ-C",
        "lczD":"LCZ-D",
        "lczG":"LCZ-G",
    }

    dot = ['digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f2f2f2"]; edge [fontsize=10];']
    for n in nodes:
        lab = label_map.get(n, n)
        fill = "#c8f7c5" if lab.endswith(final_label) or n.startswith("lcz") and lab.endswith(final_label) else "#f2f2f2"
        if n.startswith("lcz"):
            fill = "#c8f7c5" if lab.endswith(final_label) else "#ffe0b2"
        dot.append(f'"{n}" [label="{lab}", fillcolor="{fill}"];')
    for a,b,lbl in edges:
        style = "bold" if b.startswith("lcz") else "solid"
        dot.append(f'"{a}" -> "{b}" [label="{lbl}", penwidth=2, style={style}];')
    dot.append("}")
    return "\n".join(dot)



############# The UI

st.title("LCZ Classifier")
st.caption("LCZ classification, decision paths, and alternative matches")

def _lcz_sort_key(code: str):
    try:
        return (0, int(code))
    except ValueError:
        return (1, code)



with st.sidebar:
    st.header("Inputs")
    init_defaults()

    preset_options = ["— choose —"] + sorted(list(LCZ_TABLE.keys()), key=_lcz_sort_key)
    preset = st.selectbox("LCZ preset", options=preset_options, key="preset_lcz")

    if preset != "— choose —":
        if st.session_state.get("_last_preset") != preset:
            apply_preset_to_session(preset)
            st.session_state["_last_preset"] = preset
            st.rerun()

    SVF = st.number_input("SVF", min_value=0.0, max_value=1.0, step=0.01, key="SVF")
    st.caption("min = 0, max = 1")
    SCR = st.number_input("SCR", min_value=0.0, max_value=3.0, step=0.01, key="SCR")
    st.caption("min = 0, max = 3")
    FAR = st.number_input("FAR", min_value=0.0, max_value=5.0, step=0.05, key="FAR")
    st.caption("min = 0, max = 5")
    BSF = st.number_input("BSF (%)", min_value=0.0, max_value=100.0, step=1.0, key="BSF")
    st.caption("min = 0, max = 100")
    ISF = st.number_input("ISF (%)", min_value=0.0, max_value=100.0, step=1.0, key="ISF")
    st.caption("min = 0, max = 100")
    PSF = st.number_input("PSF (%)", min_value=0.0, max_value=100.0, step=1.0, key="PSF")
    st.caption("min = 0, max = 100")
    BH  = st.number_input("BH (m)", min_value=0.0, max_value=50.0, step=0.5, key="BH")
    st.caption("min = 0, max = 50")
    BHD = st.number_input("BHD: Building Height Deviation (m)", min_value=0.0, max_value=100.0, step=1.0, key="BHD")
    st.caption("min = 0, max = 100")
    BHV = st.number_input("BHV: Building Height Variance (m²)", min_value=0.0, max_value=5000.0, step=100.0, key="BHV")
    st.caption("min = 0, max = 5000")
    AL  = st.number_input("AL", min_value=0.0, max_value=0.3, step=0.01, key="AL")
    st.caption("min = 0, max = 0.3")
    TR  = st.number_input("TR", min_value=0.0, max_value=8.0, step=1.0, key="TR")
    st.caption("min = 0, max = 8")
    TH  = st.text_input("TH (text, e.g., 'Dense tree area')", key="TH")
    st.caption("Dense Tree Area, Sparse Tree Area, Dense Bush Area, Ground Cover Area, Water")




params = {p: st.session_state.get(p) for p in
          ["SVF","SCR","FAR","BSF","ISF","PSF","BH","BHD","BHV","AL","TR","TH"]}




res = classify_lcz(params)
lcz = res["lcz"]
trace = res["trace"]
nodes = res["nodes"]
edges = res["edges"]
alts = res["alternatives"]

PARAM_ORDER = ["SVF","SCR","FAR","BSF","ISF","PSF","BH","BHD","BHV","AL","TH","TR"]

def _lcz_group(code: str) -> str:
    return "Built-up" if code and code[0].isdigit() else "Land cover"

def _ranges_df(ranges_dict: dict) -> pd.DataFrame:
    data = []
    for p in PARAM_ORDER:
        v = None if ranges_dict is None else ranges_dict.get(p, "")
        data.append({"Parameter": p, "Range": v if v is not None else ""})
    return pd.DataFrame(data)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Classifier", "LCZs", "Parameters", "GIS vs WUDAPT", "District Level", "Comparison"])

with tab1:
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader(f"Result: **LCZ-{lcz}**")
        st.write(f"**Type:** {res.get('type','')}")
        st.info(res.get("definition",""))
        st.write("**Decision trace:**")
        for t in trace:
            st.markdown(f"- {t}")
        #st.code(json.dumps(params, indent=2), language="json")


    with col2:

        if res["image"]:
            if res["image"].startswith("http"):
                st.image(res["image"], caption=f"Example for LCZ-{lcz}", use_column_width=True)
            else:
                st.image(res["image"], caption=f"Example for LCZ-{lcz}", use_column_width=True)

        st.subheader("Alternative matches")
        if alts:
            alt_rows = [{"LCZ": a[0], "Hits": a[1], "Distance": round(a[2], 3)} for a in alts]
            st.table(alt_rows)
        else:
            st.write("No alternatives found.")

    st.subheader("Decision Path Visualization")
    dot = to_dot(nodes, edges, final_label=lcz)
    st.graphviz_chart(dot, use_container_width=True)


with tab2:
    st.header("LCZ info")

    if "LCZ_DEFINITIONS" not in globals() or not isinstance(LCZ_DEFINITIONS, dict) or len(LCZ_DEFINITIONS) == 0:
        st.warning("`LCZ_DEFINITIONS` not found or empty. Define it before using this tab.")
        st.stop()

    images_dict = globals().get("LCZ_IMAGES", {})

    colc1, colc2, colc3, colc4 = st.columns([1.2, 1, 1, 1])
    with colc1:
        q = st.text_input("Search (code, type, text)", "")
    with colc2:
        group_filter = st.multiselect(
            "Group",
            ["Built-up", "Land cover", "All"],
            default=["All"]
        )
    with colc3:
        show_images = st.checkbox("Show images (if available)", value=True)
    with colc4:
        compact = st.checkbox("Compact tables", value=True, help="Smaller font & height for ranges table")

    st.divider()
    df_summary = pd.DataFrame(LCZ_SUMMARY).set_index("LCZ")

    items = []
    for code, meta in LCZ_DEFINITIONS.items():
        implications = None
        if code in df_summary.index:
            row = df_summary.loc[code]
            implications = {
                "Meteorology": row["Meteorology"],
                "Social": row["Social"]
            }        

        items.append({
            "code": str(code),
            "type": meta.get("type","").strip(),
            "definition": meta.get("definition","").strip(),
            "implications": implications,
            "group": _lcz_group(str(code)),
            "image": images_dict.get(str(code)),
            "ranges": LCZ_TABLE.get(code, {})
        })

    q_lower = q.lower().strip()
    filtered = []
    all = []
    for it in items:
        if q_lower:
            hay = f"{it['code']} {it['type']} {it['definition']}".lower()
            if q_lower not in hay:
                continue
        all.append(it)
        if it["group"] not in group_filter:
            continue
        if q_lower:
            hay = f"{it['code']} {it['type']} {it['definition']}".lower()
            if q_lower not in hay:
                continue
        filtered.append(it)


    filtered = sorted(filtered, key=lambda x: _lcz_sort_key(x["code"]))
    all = sorted(all, key=lambda x: _lcz_sort_key(x["code"]))

    total = len(filtered)
    built = sum(1 for x in filtered if x["group"] == "Built-up")
    land  = total - built
    st.caption(f"Showing {total} LCZs — {built} built-up, {land} land cover")

    if not filtered:
        cols = st.columns(3)
        for i, it in enumerate(all):
            with cols[i % 3]:
                st.markdown(f"### LCZ-{it['code']}")
                st.markdown(f"**Type:** {it['type']}")
                if show_images and it["image"]:
                    st.image(it["image"], use_column_width=True, caption=f"Example — LCZ-{it['code']}")
                st.markdown(it["definition"] or "_No definition provided._")

                if it["implications"]:
                    st.markdown("**Meteorological Impact:**")
                    st.write(it["implications"]["Meteorology"])
                    st.markdown("**Social Impact:**")
                    st.write(it["implications"]["Social"])


                # Ranges table
                df_ranges = _ranges_df(it["ranges"])
                if compact:
                    st.dataframe(
                        df_ranges,
                        use_container_width=True,
                        height=280,
                        hide_index=True
                    )
                else:
                    st.table(df_ranges)

    else:
        cols = st.columns(3)
        for i, it in enumerate(filtered):
            with cols[i % 3]:
                st.markdown(f"### LCZ-{it['code']}")
                st.markdown(f"**Type:** {it['type']}")
                if show_images and it["image"]:
                    st.image(it["image"], use_column_width=True, caption=f"Example — LCZ-{it['code']}")
                st.markdown(it["definition"] or "_No definition provided._")
                #with st.expander("More"):
                #    st.write(f"Group: `{it['group']}`")
                #    if it["image"]:
                #        st.write(f"Image path/URL: `{it['image']}`")
                # Ranges table

                if it["implications"]:
                    st.markdown("**Meteorological Impact:**")
                    st.write(it["implications"]["Meteorology"])
                    st.markdown("**Social Impact:**")
                    st.write(it["implications"]["Social"])


                df_ranges = _ranges_df(it["ranges"])
                if compact:
                    st.dataframe(
                        df_ranges,
                        use_container_width=True,
                        height=280,
                        hide_index=True
                    )
                else:
                    st.table(df_ranges)


    st.divider()

    if st.checkbox("Show as table", value=False):
        df_defs = pd.DataFrame(all)[["code","type","group","definition","image"]]
        st.dataframe(df_defs, use_container_width=True)
        csv = df_defs.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="lcz_definitions_filtered.csv", mime="text/csv")




with tab3:
    st.header("Parameters")

    with st.expander("SVF (Sky View Factor)"):
        st.markdown("""
        **Definition**  
        Fraction of the sky hemisphere visible from the ground (0 = blocked, 1 = open).
        """)
        st.latex(r"\psi_{SVF} = 1 - \sum_i \sin^2(\beta_i)\,\frac{\alpha_i}{360^\circ}")
        st.markdown("""
        **Calculation**  
        - Divide horizon into slices 
        - find maximum skyline angle βᵢ for each i
        - Compute blocked fraction                    
        - Subtract from 1 for visible-sky fraction.  

        **Interpretation**  
        - Low (<0.3): Dense tall buildings, little visible sky.  
        - Medium (0.3–0.6): Moderate density.  
        - High (>0.6): Open areas, suburbs, fields.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "svf.png"), caption="Interpolated Sky View Factor", width=500)
        st.markdown("""
        **Result Interpretation**
        SVF is lower in the eastern town (more of the core) than the west
        Even though the core is mainly dense low-rise buildings, it sees a low SVF due to higher built area.
                    
                    """)

    with st.expander("SCR (Street Canyon Ratio)"):
        st.markdown("**Definition**: Ratio of average building height (H) to average street width (W).")
        st.latex(r"SCR = \frac{H}{W}")
        st.markdown("""
        **Calculation**  
        - H = mean building height along street segment.  
        - W = mean street width.  

        **Interpretation**  
        - <0.5: Wide streets, open canyons, good ventilation.  
        - ≈1: Balanced canyons (H ≈ W).  
        - \>2: Deep, narrow canyons, poor ventilation.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "scr.png"), caption="Street Canyon Ratio", width=500)
        st.markdown("""
        **Result Interpretation**
        There is about an equal distribution of street cannons
        - About 40\% are shallow street cannons, 32\% standard, and 28\% deep cannons.
        - Probably because Changsha often has really wide roads 
                    """)

    with st.expander("FAR (Floor Area Ratio)"):
        st.markdown("**Definition**: Ratio of total building floor area to plot area.")
        st.latex(r"FAR = \frac{\text{Total Floor Area}}{\text{Plot Area}}")
        st.markdown("""
        **Calculation**  
        - Floor area = footprint × floors.  
        - Divide by plot/grid cell area.  

        **Interpretation**  
        - <0.5: Low density (suburban/rural).  
        - 0.5–2: Moderate density.  
        - \>3: High-rise, very dense urban core.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "far.png"), caption="Floor Area Ratio", width=500)

    with st.expander("BSF (Building Surface Fraction)"):
        st.markdown("**Definition**: Fraction of grid cell covered by building footprints.")
        st.latex(r"BSF = \frac{\text{Building Footprint Area}}{\text{Grid Cell Area}}")
        st.markdown("""
        **Calculation**  
        - Building data collected during 2016, 2017 using Open Street Map    
        - Sum builfing footprint area in grid cell and divide by cell area. 

        **Interpretation**  
        - <10%: Sparse development.  
        - 20–40%: Moderate built-up.  
        - \>50%: Dense compact zones (LCZ 1–3).
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "bsf2.png"), caption="Building Surface Fraction", width=500)
        st.markdown("""
        **Result Interpretation**
        - The city center is really dense, but new developments, although tall, are pretty open
        - Clear gradual decline from the city center
        - Areas with high built area in the eastern side were concentrated south of the Liuyang river, which had a really high FAR area
                    
                    """)

    with st.expander("ISF (Impervious Surface Fraction)"):
        st.markdown("**Definition**: Fraction of area covered by impervious materials (roads, concrete, rooftops).")
        st.latex(r"ISF = \frac{\text{Impervious Surface Area}}{\text{Grid Cell Area}}")
        st.markdown("""
        **Calculation**  
        - Derived from remote sensing (Something complicated using NDVI and NDWI and supervised classification).  

        **Interpretation**  
        Bro I have been trying to understand what they exactly mean but the plot does not make sense. Even if i see it as the fraction of area thats impervious or what they mention in the paper as area of vegetation and water by total.
        - <20%: Mostly natural cover.  
        - 40–60%: Mixed cover.  
        - \>70%: Heavily sealed urban core.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "isf.png"), caption="Impervious Surface Fraction", width=500)
        st.markdown("""
        **Result Interpretation**
        - Highly impervious regions were mainly the railway station, some markets, and industrian parks and economic zones.
        - Which were like urban paved areas with car parks, buildings, and roads...
        - Furong in the east had high density commecrial centers and residential land with high ISF
                    """)
        
    with st.expander("PSF (Pervious Surface Fraction)"):
        st.markdown("**Definition**: Fraction of grid cell covered by permeable ground (vegetation, soil).")
        st.latex(r"PSF = \frac{\text{Pervious Surface Area}}{\text{Grid Cell Area}}")
        st.markdown("""
        **Calculation**  
        - Typically = 1 − ISF (excluding water bodies).  
        - Using vegetation indices (NDVI, NDWI).  

        **Interpretation**  
        - <20%: Dense urban cores.  
        - 40–70%: Suburban/peri-urban.  
        - \>80%: Natural/green LCZs.
                    
        **Result Interpretation**
        Areas with a lower population generally were more pervious, mostly in the western side of the city
        """)
        #st.image(f"{CUR_DIR}/images/parimages/ndvi.png", caption="NDVI", width=500)

    with st.expander("BH (Mean Building Height)"):
        st.markdown("**Definition**: Average building height in the grid.")
        st.latex(r"BH = \frac{\sum_i h_i}{N}")
        st.markdown("""
        **Calculation**  
        - Fisheye Cameras
        - Building data collected during 2016, 2017 using Open Street Map  

        **Interpretation**  
        - <9 m: Low-rise zones.  
        - 10–25 m: Mid-rise.  
        - \>25 m: High-rise.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "bh.png"), caption="Building Heights", width=500)
        st.markdown("""
        **Result Interpretation**
        - Building heights gradually decrease from city center to the surrounding areas.
        - Central areas east of the river have a large area of mid-rise buildings
        - However Kaifu and Yuhua districts have the majority of high rise buildings
                    """)

    with st.expander("BHD (Building Height Deviation)"):
        st.markdown("**Definition**: Mean absolute deviation of building heights within the grid (m).")
        st.latex(r"\text{BHD} = \frac{1}{N}\sum_{i=1}^{N} \left| h_i - \overline{h} \right|")
        st.markdown("""
        **Calculation**
        - $h_i$: height of building $i$;  $\overline{h}$: mean height.
        - Building data collected during 2016, 2017 using Open Street Map 
                    
        **Interpretation**
        - Low BHD → more uniform roofline; High BHD → mixed/irregular heights.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "bhd.png"), caption="Building Height Deviation", width=500)
        st.markdown("""
        **Result Interpretation**
        The suburbs have high BHV and BHD due to new high rise residential areas around the original scattered low rise buildings. 
                   """)

    with st.expander("BHV (Building Height Variance)"):
        st.markdown("**Definition**: Variance of building heights within the grid (m²).")
        st.latex(r"\text{BHV} = \frac{1}{N}\sum_{i=1}^{N} \left(h_i - \overline{h}\right)^2")
        st.markdown("""
        **Calculation**
        - Building data collected during 2016, 2017 using Open Street Map 
                    
        **Interpretation**
        - High BHV indicates strong mixing of low- and high-rise buildings.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "bhv.png"), caption="Building Height Deviation", width=500)
        st.markdown("""
        **Result Interpretation**
        The suburbs have high BHV and BHD due to new high rise residential areas around the original scattered low rise buildings. 
                   """)

    with st.expander("AL (Albedo)"):
        st.markdown("**Definition**: Surface reflectivity (ratio of reflected to incoming shortwave radiation).")
        st.latex(r"Albedo = \frac{R_{SW}^{\uparrow}}{R_{SW}^{\downarrow}}")
        st.markdown("""
        **Calculation**  
        - Derived from remote sensing data (They used Landsat 8)

        **Interpretation**  
        - 0.05–0.15: Asphalt, dark roofs.  
        - 0.15–0.3: Concrete, stone.  
        - \>0.3: Bright reflective roofs/surfaces.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "al.png"), caption="Albedo", width=500)
        st.markdown("""
        **Result Interpretation**
        Albedo was generally higher in the newly developed regions
        - Probably because of glass buildings 
                   """)

    with st.expander("TR (Terrain Roughness)"):
        st.markdown("""
        **Definition**  
        Measure of aerodynamic drag caused by surface obstacles (buildings, vegetation).  
        Usually parameterized as a roughness length (z₀).
        8 levels from the smooth sea to chaotic.
        The sea will have a roughness length of 0.0002m, vs large obstacles in urban areas would have a roughness length of >2m.
        (z₀) represents the physical effect of surface obstacles on the wind velocity profile. 
        """)
        st.markdown("**Calculation**: Based on building height/spacing distribution. They've used supervised classification for Davenport's 2000 methodology and I do not have access to that paper.")
        st.markdown("""
        **Interpretation**  
        - 8: Dense urban core, Large Forest.  
        - 5–6: Suburban or tree-dominated.  
        - 3–4: Low vegetation/open land.  
        - 1: Water, very smooth.
        """)
        st.image(os.path.join(CUR_DIR, "images", "parimages", "tr.png"), caption="Terrain Roughness", width=500)

    with st.expander("TH (Tree / Cover Type)"):
        st.markdown("""
        **Definition**  
        Vegetation/cover type (dense trees, scattered trees, shrubs, ground cover, water).
        """)
        st.markdown("**Calculation**: Remote sensing classification (NDVI and NDWI). Supervised Classification")
        st.markdown("""
        **Interpretation**  
        - Dense trees → LCZ-A  
        - Scattered trees → LCZ-B  
        - Bush/scrub → LCZ-C  
        - Low plants/grass → LCZ-D  
        - Bare rock/soil/water → LCZ-E/F/G
        """)

        st.image(os.path.join(CUR_DIR, "images", "parimages", "th.png"), caption="Tree Categorisation", width=500)
        st.markdown("""
        **Result Interpretation**
        Most land cover was dominated by Dense forests and Open ground
        - Dense forests were about 16\% of total, while open ground was ~27
                   """)

PLOTS_DIR = os.path.join(CUR_DIR, "images", "plots")
SUFFIXES = {
    "Overall Accuracy": "_proportions.png",
    "Proportions": "_proportions_by_lcz.png",
    "Stacked by LCZ": "_stacked_by_lcz.png",
    "Stacked by Main LCZ group": "_stacked_by_group.png",
    "UA & PA": "_uapa_grouped.png",
}

def find_districts_from_files(folder=PLOTS_DIR):
    if not os.path.isdir(folder):
        return []
    names = set()
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".png"):
            continue
        for suf in SUFFIXES.values():
            if fn.endswith(suf):
                names.add(fn[: -len(suf)])
    return sorted(names)

def load_image(path):
    try:
        return Image.open(path)
    except Exception:
        return None

def show_image_with_download(title, path):
    st.markdown(f"**{title}**")
    img = load_image(path)
    if img is None:
        st.warning(f"Missing: `{path}`")
        return
    st.image(img, width=900)
    with open(path, "rb") as f:
        st.download_button(
            label=f"Download PNG: {os.path.basename(path)}",
            data=f.read(),
            file_name=os.path.basename(path),
            mime="image/png",
            key=path,  
        )

with tab4:
    st.header("Overall GIS vs WUDAPT Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final GIS based LCZ classification")
        st.image(os.path.join(CUR_DIR, "images", "gislcz.png"), width=650, use_column_width=False)

    with col2:
        st.subheader("Final WUDAPT based LCZ classification")
        st.image(os.path.join(CUR_DIR, "images", "WUDAPTlcz.png"), width=615, use_column_width=False)

    st.divider()

    # Summary text
    st.markdown("""
    WUDAPT had an overall accuracy of **58.72%** with a kappa coefficient of **0.54**  
    (where -1 = complete disagreement, 0 = random chance, 1 = perfect agreement).  

    GIS-based classification had an overall accuracy of **84.4%**.  

    - WUDAPT tends to simplify complex and irregular urban structures. 
    - Mainly classified urban areas are LCZ-5 
    - Both are alright for a city scale to get the built area, land cover, and water classified right
    - But the overall accuracy of GIS is better for district level classification

    """)


with tab5:
    st.header("District Level")

    districts = find_districts_from_files(PLOTS_DIR)
    if not districts:
        st.error(f"No PNGs found in `{PLOTS_DIR}/`. Expected files like `Yuelu_stacked_by_lcz.png`.")
    else:
        sel = st.selectbox("Select district", districts)

        st.write("**Choose plots to display**")
        cols = st.columns(2)
        with cols[0]:
            show_stacked_lcz = st.checkbox("Stacked by LCZ)", value=True)
            show_stacked_group = st.checkbox("Stacked by Main LCZ group", value=True)
            show_proportions = st.checkbox("Proportions", value=False)
        with cols[1]:
            show_uapa_grouped = st.checkbox("UA & PA", value=False)
            show_overall_accuracy = st.checkbox("Overall Accuracy", value=False)

        st.divider()

        if show_stacked_lcz:
            p = os.path.join(PLOTS_DIR, f"{sel}{SUFFIXES['Stacked by LCZ']}")
            show_image_with_download("Stacked by LCZ (per method)", p)

        if show_stacked_group:
            p = os.path.join(PLOTS_DIR, f"{sel}{SUFFIXES['Stacked by Main LCZ group']}")
            show_image_with_download("Stacked by MainLCZ group", p)

        if show_proportions:
            p = os.path.join(PLOTS_DIR, f"{sel}{SUFFIXES['Proportions']}")
            show_image_with_download("Proportions", p)

        if show_uapa_grouped:
            p = os.path.join(PLOTS_DIR, f"{sel}{SUFFIXES['UA & PA']}")
            show_image_with_download("UA & PA grouped", p)

        if show_overall_accuracy:
            p = os.path.join(PLOTS_DIR, f"{sel}{SUFFIXES['Overall Accuracy']}")
            show_image_with_download("Overall Accuracy", p)


        with st.expander("See all available images for this district"):
            thumbs = [
                (label, os.path.join(PLOTS_DIR, f"{sel}{suf}"))
                for label, suf in SUFFIXES.items()
                if os.path.exists(os.path.join(PLOTS_DIR, f"{sel}{suf}"))
            ]
            if not thumbs:
                st.info("No images found for this district.")
            else:
                ncols = min(3, len(thumbs))
                grid = st.columns(ncols)
                for i, (label, path) in enumerate(thumbs):
                    with grid[i % ncols]:
                        img = load_image(path)
                        if img:
                            st.image(img, caption=label, use_column_width=True)

 


COMP_DIR = os.path.join(CUR_DIR, "images", "punevchangsha")

with tab6:
    st.header("City Comparison: Pune vs Changsha")

    st.markdown("""
    ### Overview
    The study area of Changsha here is about 352 km$^2$. It has a metro population of about 10 million. 99.2% of them are han chinese its crazy. \\
    The study area of Pune here is about 243 km$^2$, although its now 516 (up from 331), with a metro population of ~7.4 million.
                
    Despite having similar sizes, the GDP of Changsha is about 4 times higher at \$220 Billion.
    Mao was lowkey radicalised in Changsha, and its a culturally important place for its western Han dynasty tombs. 

    This tab compares LCZ classification outcomes for both cities.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pune Results")
        pune_img = os.path.join(COMP_DIR, "punelcz.png")  
        if os.path.exists(pune_img):
            st.image(pune_img, caption="Pune WUDAPT LCZ Classification. Credit: Prasad Pathak", use_column_width=True)
        else:
            st.info("Pune comparison plot not found.")

        st.markdown("""
        - Pune mostly dominated by LCZ5, followed by open mid-rise
        - Some open and bare area toward the north-eastern periphery (LCZ F)
        - Lots of dense forests near the pashan and tekdi area
        - But not many greenspacees within the built up regions
        - About 82% accurate
        """)

    with col2:
        st.subheader("Changsha Results")
        changsha_img = os.path.join(COMP_DIR, "changshalcz.png")   
        if os.path.exists(changsha_img):
            st.image(changsha_img, caption="Changsha WUDAPT LCZ Classification", use_column_width=True)
        else:
            st.info("Changsha comparison plot not found.")

        st.markdown("""
        - Most of the city area is still classified under non-built up land cover at 57%.
        - LCZ5 was still the most common built up LCZ, but barely any open low-rise (LCZ 6)
        - High rises mostly common around major roads and newly built areas. Same as viman nagar.
        - Overall accuracy of (the map) about 58%. (but the text is from the high accuracy one)
        """)



