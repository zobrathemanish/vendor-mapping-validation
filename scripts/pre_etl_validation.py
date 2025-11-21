import os
import json
from typing import Dict, Any, List

import pandas as pd
import yaml

# -------------------------------
# CONFIG
# -------------------------------
MAPPED_DIR = "mapped"                  # Output from mapping
SILVER_BASE_DIR = os.path.join("silver", "in_review")
VALIDATION_CONFIG_DIR = "validation_config"    # NEW external config folder


# ================================================================
# 1. LOAD VALIDATION CONFIG (EXTERNAL YAML)
# ================================================================
def load_validation_config(vendor: str) -> Dict[str, Any] | None:
    """
    Looks for:
       validation_config/<vendor_lower>.yaml
    Vendor folder uses vendor=Grote Lighting,
    so YAML file is: grote lighting.yaml
    """
    # Match YAML filename format
    fname = vendor.lower().strip() + ".yaml"
    path = os.path.join(VALIDATION_CONFIG_DIR, fname)

    if not os.path.exists(path):
        print(f"⚠️  No validation config YAML found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ================================================================
# 2. DISCOVER VENDORS FROM MAPPED/
# ================================================================
def discover_mapped_vendors() -> List[str]:
    if not os.path.exists(MAPPED_DIR):
        return []

    vendors = []
    for name in os.listdir(MAPPED_DIR):
        if name.startswith("vendor="):
            vendors.append(name.split("=", 1)[1])
    return sorted(vendors)


# ================================================================
# 3. ITEM MASTER VALIDATION
# ================================================================
def validate_item_master(vendor: str, df: pd.DataFrame, cfg: Dict[str, Any], errors: List):
    required = cfg.get("required_fields", [])
    optional = cfg.get("optional_fields", [])
    unique_key = cfg.get("unique_key", "Part Number")

    is_valid = pd.Series(True, index=df.index)

    # Required fields check
    for field in required:
        if field not in df.columns:
            for idx, row in df.iterrows():
                errors.append({
                    "Vendor": vendor, "Section": "Item Master",
                    "Part Number": row.get("Part Number"),
                    "Field": field, "ErrorCode": "MISSING_COLUMN",
                    "Message": f"Required column '{field}' missing."
                })
            is_valid[:] = False
            continue

        missing = df[field].isna() | (df[field].astype(str).str.strip() == "")
        for idx in df[missing].index:
            errors.append({
                "Vendor": vendor, "Section": "Item Master",
                "Part Number": df.at[idx, "Part Number"],
                "Field": field, "ErrorCode": "REQUIRED_FIELD_EMPTY",
                "Message": f"Required field '{field}' is empty."
            })
        is_valid[missing] = False

    # Optional fields → no errors, only warnings (future)
    # Duplicate key check
    if unique_key in df.columns:
        dup_mask = df[unique_key].duplicated(keep=False)
        for idx in df[dup_mask].index:
            errors.append({
                "Vendor": vendor, "Section": "Item Master",
                "Part Number": df.at[idx, unique_key],
                "Field": unique_key,
                "ErrorCode": "DUPLICATE_KEY",
                "Message": f"Duplicate Part Number '{df.at[idx, unique_key]}'."
            })
        is_valid[dup_mask] = False

    return is_valid


# ================================================================
# 4. DESCRIPTIONS VALIDATION
# ================================================================
def validate_descriptions(vendor, item_master, df_desc, cfg, errors):

    if df_desc is None or df_desc.empty:
        if cfg.get("require_des_for_all_skus", False):
            for _, row in item_master.iterrows():
                errors.append({
                    "Vendor": vendor, "Section": "Descriptions",
                    "Part Number": row.get("Part Number"),
                    "Field": "Description Value",
                    "ErrorCode": "MISSING_DES",
                    "Message": "Missing DES description."
                })
        return pd.Series(False, index=pd.Index([]))

    code_field = cfg["code_field"]
    value_field = cfg["value_field"]
    seq_field = cfg["sequence_field"]
    max_len_codes = cfg.get("max_length_codes", {})

    is_valid = pd.Series(True, index=df_desc.index)

    # Length checks
    for code, max_len in max_len_codes.items():
        mask = df_desc[code_field] == code
        too_long = mask & df_desc[value_field].astype(str).str.len().gt(max_len)

        for idx in df_desc[too_long].index:
            errors.append({
                "Vendor": vendor, "Section": "Descriptions",
                "Part Number": df_desc.at[idx, "Part Number"],
                "Field": value_field,
                "ErrorCode": "DESC_TOO_LONG",
                "Message": f"{code} exceeds {max_len} characters."
            })
        is_valid[too_long] = False

    # Sequence numeric check
    non_numeric_seq = (
        df_desc[seq_field].notna()
        & ~df_desc[seq_field].astype(str).str.isdigit()
    )
    for idx in df_desc[non_numeric_seq].index:
        errors.append({
            "Vendor": vendor, "Section": "Descriptions",
            "Part Number": df_desc.at[idx, "Part Number"],
            "Field": seq_field,
            "ErrorCode": "SEQ_NOT_NUMERIC",
            "Message": "Sequence must be numeric."
        })
    is_valid[non_numeric_seq] = False

    # Missing DES per SKU
    if cfg.get("require_des_for_all_skus", False):
        des_skus = set(df_desc[df_desc[code_field] == "DES"]["Part Number"].dropna().unique())
        for sku in item_master["Part Number"]:
            if sku not in des_skus:
                errors.append({
                    "Vendor": vendor, "Section": "Descriptions",
                    "Part Number": sku,
                    "Field": "Description Value",
                    "ErrorCode": "MISSING_DES",
                    "Message": "Missing DES description."
                })

    return is_valid


# ================================================================
# 5. PRICING VALIDATION (WITH CLEANING)
# ================================================================
NUMERIC_PRICE_FIELDS = [
    "List Price",
    "Jobber Price",
    "Pricing Amount (Net)",
    "Dealer Price",
    "Discount %",
]

def clean_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes $, commas, extracts numeric discount, converts to float.
    """
    if df is None or df.empty:
        return df

    # Remove $ and commas
    for col in NUMERIC_PRICE_FIELDS:
        df[col] = df[col].apply(clean_price_generic)


    # Extract numeric Discount %
    if "Discount %" in df.columns:
        df["Discount %"] = (
            df["Discount %"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # Convert numeric fields
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def validate_pricing(vendor, item_master, df_price, cfg, errors):

    if df_price is None or df_price.empty:
        if cfg.get("require_pricing_for_all_skus", False):
            for sku in item_master["Part Number"]:
                errors.append({
                    "Vendor": vendor, "Section": "Pricing",
                    "Part Number": sku,
                    "Field": "Part Number",
                    "ErrorCode": "MISSING_PRICING",
                    "Message": "Missing pricing record."
                })
        return pd.Series(False, index=pd.Index([]))

    df_price = clean_pricing(df_price)

    required = cfg.get("required_fields", [])
    numeric_positive = cfg.get("numeric_positive_fields", [])
    numeric_fields = cfg.get("numeric_fields", [])

    is_valid = pd.Series(True, index=df_price.index)

    # Required fields
    for field in required:
        missing = df_price[field].isna() | (df_price[field].astype(str).str.strip() == "")
        for idx in df_price[missing].index:
            errors.append({
                "Vendor": vendor, "Section": "Pricing",
                "Part Number": df_price.at[idx, "Part Number"],
                "Field": field,
                "ErrorCode": "REQUIRED_FIELD_EMPTY",
                "Message": f"Missing required pricing field '{field}'."
            })
        is_valid[missing] = False

    # Numeric positive
    for field in numeric_positive:
        vals = pd.to_numeric(df_price[field], errors="coerce")
        invalid = vals.isna() | (vals <= 0)
        idxs = df_price[invalid].index
        for idx in idxs:
            errors.append({
                "Vendor": vendor, "Section": "Pricing",
                "Part Number": df_price.at[idx, "Part Number"],
                "Field": field,
                "ErrorCode": "NON_POSITIVE_VALUE",
                "Message": f"Field '{field}' must be >0."
            })
        is_valid[idxs] = False

    # Generic numeric validation
    for field in numeric_fields:
        vals = pd.to_numeric(df_price[field], errors="coerce")
        nonnum = vals.isna() & df_price[field].notna()
        for idx in df_price[nonnum].index:
            errors.append({
                "Vendor": vendor, "Section": "Pricing",
                "Part Number": df_price.at[idx, "Part Number"],
                "Field": field,
                "ErrorCode": "NOT_NUMERIC",
                "Message": f"Field '{field}' must be numeric."
            })
        is_valid[nonnum] = False

    return is_valid

import re

def clean_price_generic(val):
    """
    Cleans any vendor price format into a float.
    Works for: $1,200.50 | CAD 1200 | 1200 USD | Net: $2500 | ' 1,200 ' etc.
    Returns float or None.
    """
    if val is None:
        return None
    
    # Allow numbers, commas, periods inside long vendor text
    text = str(val)

    # Remove currency codes & symbols
    text = re.sub(r'[^\d.,-]', '', text)  # keep digits, . , and -

    # Remove commas
    text = text.replace(",", "")

    # If empty after cleaning → no numeric value
    if text.strip() == "":
        return None

    # Convert to float
    try:
        return float(text)
    except:
        return None



# ================================================================
# 6. DIGITAL ASSETS VALIDATION
# ================================================================
def validate_assets(vendor, item_master, df_assets, cfg, errors):

    if df_assets is None or df_assets.empty:
        if cfg.get("require_asset_for_all_skus", False):
            for sku in item_master["Part Number"]:
                errors.append({
                    "Vendor": vendor, "Section": "Digital Assets",
                    "Part Number": sku,
                    "Field": "FileName",
                    "ErrorCode": "MISSING_ASSET",
                    "Message": "No digital assets found."
                })
        return pd.Series(False, index=pd.Index([]))

    is_valid = pd.Series(True, index=df_assets.index)

    if cfg.get("require_asset_for_all_skus", False):
        asset_skus = set(df_assets["Part Number"].dropna().unique())
        for sku in item_master["Part Number"]:
            if sku not in asset_skus:
                errors.append({
                    "Vendor": vendor, "Section": "Digital Assets",
                    "Part Number": sku,
                    "Field": "FileName",
                    "ErrorCode": "MISSING_ASSET",
                    "Message": "No digital assets found."
                })

    return is_valid


# ================================================================
# 7. PER-VENDOR VALIDATION PIPELINE
# ================================================================
def process_vendor(vendor: str):

    print(f"\n🔵 Running validation for vendor: {vendor}")

    # Load validation YAML
    cfg_file = load_validation_config(vendor)
    if not cfg_file:
        print(f"⚠️ Skipping vendor '{vendor}' — no YAML config.")
        return

    cfg = cfg_file["validation"]

    mapped_dir = os.path.join(MAPPED_DIR, f"vendor={vendor}")
    if not os.path.exists(mapped_dir):
        print(f"⚠️ No mapped data: {mapped_dir}")
        return

    # Load parquets
    def load_parquet(name: str):
        path = os.path.join(mapped_dir, f"{name}.parquet")
        return pd.read_parquet(path) if os.path.exists(path) else None

    df_item     = load_parquet("Item_Master")
    df_desc     = load_parquet("Descriptions")
    df_price    = load_parquet("Pricing")
    df_assets   = load_parquet("Digital_Assets")

    if df_item is None or df_item.empty:
        print("❌ Missing Item_Master.parquet. Cannot validate.")
        return

    errors = []

    # Run individual validators
    item_valid   = validate_item_master(vendor, df_item, cfg["item_master"], errors)
    desc_valid   = validate_descriptions(vendor, df_item, df_desc, cfg["descriptions"], errors) if df_desc is not None else None
    price_valid  = validate_pricing(vendor, df_item, df_price, cfg["pricing"], errors) if df_price is not None else None
    asset_valid  = validate_assets(vendor, df_item, df_assets, cfg["assets"], errors) if df_assets is not None else None

    # Build SKU-level flags summary
    sku_col = "Part Number"
    df_flags = df_item.copy()
    df_flags["is_item_master_valid"] = item_valid

    # descriptions
    if df_desc is not None and desc_valid is not None:
        sku_valid_desc = (
            df_desc.assign(_v=desc_valid)
                   .groupby(sku_col)["_v"]
                   .any()
        )
        df_flags["has_valid_descriptions"] = df_flags[sku_col].map(sku_valid_desc).fillna(False)
    else:
        df_flags["has_valid_descriptions"] = False

    # pricing
    if df_price is not None and price_valid is not None:
        sku_valid_price = (
            df_price.assign(_v=price_valid)
                    .groupby(sku_col)["_v"]
                    .any()
        )
        df_flags["has_pricing"] = df_flags[sku_col].isin(df_price[sku_col].unique())
        df_flags["has_valid_pricing"] = df_flags[sku_col].map(sku_valid_price).fillna(False)
    else:
        df_flags["has_pricing"] = False
        df_flags["has_valid_pricing"] = False

    # assets
    if df_assets is not None and asset_valid is not None:
        sku_has_assets = df_assets.groupby(sku_col).size() > 0
        df_flags["has_assets"] = df_flags[sku_col].map(sku_has_assets).fillna(False)
    else:
        df_flags["has_assets"] = False

    # overall validity
    df_flags["is_overall_valid"] = (
        df_flags["is_item_master_valid"]
        & df_flags["has_valid_descriptions"]
        & df_flags["has_valid_pricing"]
        & df_flags["has_assets"]
    )

    # ===============================
    # WRITE OUTPUTS TO SILVER
    # ===============================
    vendor_silver = os.path.join(SILVER_BASE_DIR, f"vendor={vendor}")
    os.makedirs(vendor_silver, exist_ok=True)

    df_flags.to_parquet(os.path.join(vendor_silver, "item_master_with_validation.parquet"), index=False)
    print(f"  ✅ Saved item_master_with_validation.parquet")

    if errors:
        df_err = pd.DataFrame(errors)
        # Force columns to strings for PyArrow compatibility
        for col in ["Part Number", "Field", "Vendor", "Section", "ErrorCode", "Message"]:
            if col in df_err.columns:
                df_err[col] = df_err[col].astype(str)

        df_err.to_parquet(os.path.join(vendor_silver, "validation_errors.parquet"), index=False)

        with open(os.path.join(vendor_silver, "validation_errors.json"), "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

        print("  ⚠️ Validation errors found and saved.")
    else:
        print("  🎉 No validation errors for this vendor!")


# ================================================================
# 8. ENTRYPOINT
# ================================================================
def main():
    vendors = discover_mapped_vendors()
    print("Discovered mapped vendors:", vendors)

    for v in vendors:
        process_vendor(v)


if __name__ == "__main__":
    main()
