import os
import json
import traceback
import datetime
from io import BytesIO

import yaml
import pandas as pd
from dotenv import load_dotenv
from lxml import etree
from azure.storage.blob import BlobServiceClient

load_dotenv()

# -------------------------------
# CONFIG
# -------------------------------
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "bronze"
RAW_PREFIX = "raw/vendor="  # inside the bronze container

MAPPING_OUTPUT_DIR = "mapped"
MAPPINGS_DIR = "mappings"

blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)


# ---------------------------------------------
# LOAD YAML WITH VENDOR NAME FALLBACK
# ---------------------------------------------
def load_vendor_mapping(vendor: str) -> dict | None:
    vendor_lower = vendor.lower().strip()
    candidates = [
        vendor_lower,
        vendor_lower.replace(" ", "_"),
        vendor_lower.split()[0],
        vendor_lower.split()[0].lower(),
    ]

    tried = []
    for base in dict.fromkeys(candidates):
        filename = os.path.join(MAPPINGS_DIR, f"{base}.yaml")
        tried.append(filename)
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    print(f"⚠️  No mapping YAML found for vendor '{vendor}'. Tried:", tried)
    return None


# ---------------------------------------------
# DISCOVER VENDORS
# ---------------------------------------------
def discover_vendors() -> list[str]:
    container = blob_service.get_container_client(CONTAINER_NAME)
    vendors = set()

    for blob in container.list_blobs(name_starts_with=RAW_PREFIX):
        parts = blob.name.split("/")
        for p in parts:
            if p.startswith("vendor="):
                vendors.add(p.split("=")[1])

    return sorted(vendors)


# ---------------------------------------------
# RULE-BASED LISTING (LATEST product/pricing)
# ---------------------------------------------
def list_vendor_files(vendor: str, kind: str) -> list[str]:
    prefix = f"{RAW_PREFIX}{vendor}/{kind}/"
    container = blob_service.get_container_client(CONTAINER_NAME)
    blobs = list(container.list_blobs(name_starts_with=prefix))

    if not blobs:
        return []

    # product + pricing = only latest
    if kind in ("product", "pricing"):
        blobs.sort(key=lambda b: b.last_modified, reverse=True)
        return [blobs[0].name]

    # assets + logs = all
    return [b.name for b in blobs]


# ---------------------------------------------
# DOWNLOAD BLOB BYTES
# ---------------------------------------------
def download_blob_bytes(path: str) -> bytes:
    container = blob_service.get_container_client(CONTAINER_NAME)
    blob = container.get_blob_client(path)
    return blob.download_blob().readall()


# ---------------------------------------------
# INGESTION ERROR LOGGING
# ---------------------------------------------
def log_ingestion_error(vendor: str, stage: str, file: str | None, error: Exception):
    log = {
        "vendor": vendor,
        "file": file,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc()
    }

    filename = f"{datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}_ingestion_error.json"

    # --------------------------
    # 1. Bronze log (raw truth)
    # --------------------------
    bronze_path = f"{RAW_PREFIX}{vendor}/logs/{filename}"
    blob_service.get_blob_client(CONTAINER_NAME, bronze_path).upload_blob(
        json.dumps(log), overwrite=True
    )
    print(f"⚠️ Logged ingestion error → {bronze_path}")

    # --------------------------
    # 2. Silver rejected log (feedback to vendor)
    # --------------------------
    silver_path = f"silver/rejected/logs/vendor={vendor}/{filename}"
    silver_container = blob_service.get_container_client(CONTAINER_NAME)
    silver_container.upload_blob(silver_path, json.dumps(log), overwrite=True)

    print(f"⚠️ Copied ingestion summary to → {silver_path}")



# ---------------------------------------------
# NAMESPACE NORMALIZER (KEY FIX!!)
# ---------------------------------------------
def normalize_ns(root):
    """
    Convert default namespace into 'ns' prefix.
    Required for all PIES XML.
    """
    nsmap = root.nsmap.copy()

    if None in nsmap:
        nsmap["ns"] = nsmap.pop(None)

    return nsmap


# ---------------------------------------------
# GENERIC XPATH VALUE EXTRACTOR
# ---------------------------------------------
def _extract_xpath_single(node, expr: str, nsmap):
    res = node.xpath(expr, namespaces=nsmap)
    if not res:
        return None
    val = res[0]
    if hasattr(val, "text"):
        return val.text
    return str(val)


# ---------------------------------------------
# ITEM MASTER MAPPING
# ---------------------------------------------
def map_item_master(root: etree._Element, section_cfg: dict, nsmap) -> pd.DataFrame:
    items = root.xpath(".//ns:Items/ns:Item", namespaces=nsmap)
    rows = []

    for item in items:
        row = {}
        for col, expr in section_cfg.items():
            if expr is None:
                row[col] = None
                continue

            if expr.startswith("@"):
                row[col] = item.get(expr[1:])
            elif ":" not in expr and "/" not in expr:
                node = item.find(f"ns:{expr}", namespaces=nsmap)
                row[col] = node.text if node is not None else None
            else:
                row[col] = _extract_xpath_single(item, expr, nsmap)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------
# GENERIC XML SECTION MAPPER (for Descriptions, Extended Info, etc.)
# ---------------------------------------------
def map_xml_section(root: etree._Element, path: str, mappings: dict, nsmap) -> pd.DataFrame:
    nodes = root.xpath(".//" + path, namespaces=nsmap)
    rows = []

    for node in nodes:
        row = {}
        for col, expr in mappings.items():
            if expr is None:
                row[col] = None
            elif expr == "text":
                row[col] = node.text
            elif expr.startswith("@"):
                row[col] = node.get(expr[1:])
            else:
                row[col] = _extract_xpath_single(node, expr, nsmap)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------
# PRICING (XLSX)
# ---------------------------------------------
def map_pricing_xlsx(df: pd.DataFrame, pricing_cfg: dict) -> pd.DataFrame:
    src_map = pricing_cfg.get("mappings", {})
    out = pd.DataFrame()

    for col, src in src_map.items():
        if src is None:
            out[col] = None
        else:
            if src not in df.columns:
                out[col] = src
            else:
                out[col] = df[src]

    return out


# ---------------------------------------------
# MAIN PROCESSOR
# ---------------------------------------------
def process_vendor(vendor: str):
    vendor_clean = vendor.strip()
    mapping = load_vendor_mapping(vendor_clean)

    if not mapping:
        return

    print(f"\n🔵 Processing vendor: {vendor_clean}")

    # ---------------------------
    # PRODUCT XML
    # ---------------------------
    product_files = list_vendor_files(vendor_clean, "product")
    product_sections = {}

    for blob_path in product_files:
        print(f"  • XML: {blob_path}")

        try:
            xml_bytes = download_blob_bytes(blob_path)
            root = etree.fromstring(xml_bytes)
        except Exception as e:
            print("  ❌ XML parsing failed:", e)
            log_ingestion_error(vendor_clean, "XML_PARSE", blob_path, e)
            return

        nsmap = normalize_ns(root)
        pm = mapping.get("product_mapping", {})

        # Item Master
        if "Item Master" in pm:
            df_item = map_item_master(root, pm["Item Master"], nsmap)
            product_sections.setdefault("Item Master", []).append(df_item)

        # Other XML sections
        for sec_name in ["Descriptions", "Extended Info", "Attributes", "Part InterChange", "Packages", "Digital Assets"]:
            sec_cfg = pm.get(sec_name)
            if not sec_cfg:
                continue

            path = sec_cfg.get("path")
            mappings = sec_cfg.get("mappings")
            if not path or not mappings:
                continue

            df_sec = map_xml_section(root, path, mappings, nsmap)
            product_sections.setdefault(sec_name, []).append(df_sec)

    # Combine XML results
    combined = {}
    for k, dfs in product_sections.items():
        combined[k] = pd.concat(dfs, ignore_index=True)

    # ---------------------------
    # PRICING XLSX
    # ---------------------------
    pricing_files = list_vendor_files(vendor_clean, "pricing")
    pricing_cfg = mapping.get("product_mapping", {}).get("Pricing")

    if pricing_files and pricing_cfg:
        f = pricing_files[0]
        print(f"  • Pricing: {f}")
        try:
            xbytes = download_blob_bytes(f)
            raw_df = pd.read_excel(BytesIO(xbytes))
            combined["Pricing"] = map_pricing_xlsx(raw_df, pricing_cfg)
        except Exception as e:
            print("  ❌ Pricing load failed:", e)
            log_ingestion_error(vendor_clean, "PRICING_PARSE", f, e)

    # ---------------------------
    # ASSETS & LOGS
    # ---------------------------
    assets = list_vendor_files(vendor_clean, "assets")
    logs = list_vendor_files(vendor_clean, "logs")

    if assets:
        combined["Assets_Blobs"] = pd.DataFrame({"BlobPath": assets})
    if logs:
        combined["Logs_Blobs"] = pd.DataFrame({"BlobPath": logs})

    # ---------------------------
    # POST-PROCESSING: BUILD FilePath FOR DIGITAL ASSETS
    # ---------------------------
    if "Digital Assets" in combined:
        df_assets = combined["Digital Assets"]

        # Construct blob path for mapped assets
        df_assets["FilePath"] = (
            f"raw/vendor={vendor_clean}/assets/" + df_assets["FileName"].astype(str)
        )

        combined["Digital Assets"] = df_assets
        print("  🔧 Added FilePath to Digital Assets")


    # ---------------------------
    # SAVE ALL PARQUETS
    # ---------------------------
    outdir = os.path.join(MAPPING_OUTPUT_DIR, f"vendor={vendor_clean}")
    os.makedirs(outdir, exist_ok=True)

    for name, df in combined.items():
        path = os.path.join(outdir, f"{name.replace(' ', '_')}.parquet")
        df.to_parquet(path, index=False)
        print(f"  ✅ Saved {name} → {path}")

    print(f"✅ Completed mapping for vendor {vendor_clean}")
    


# ---------------------------------------------
# ENTRYPOINT
# ---------------------------------------------
def main():
    vendors = discover_vendors()
    print("Discovered vendors:", vendors)
    for v in vendors:
        process_vendor(v)


if __name__ == "__main__":
    main()
