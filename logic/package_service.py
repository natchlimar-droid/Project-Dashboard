# -*- coding: utf-8 -*-
"""
Package Service
Handles merging real sales data with the Package Catalog, sorting by VN,
and identifying mismatches.
"""

import copy
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from logic.config import PACKAGE_CATALOG, PACKAGE_KEYWORD_ALIASES, CATEGORY_ORDER, is_excluded_package


def compute_package_stats(
    sales_df: Optional[pd.DataFrame],
    packages_dict: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Groups sales_df by package_name to sum total_vn and total_revenue,
    merges stats into packages_dict, sorts packages inside each category
    by total_vn descending, and tracks any unmatched packages.

    Returns:
        (updated_packages_dict, mismatches_list)
    """
    if packages_dict is None:
        pkgs = copy.deepcopy(PACKAGE_CATALOG)
    else:
        pkgs = copy.deepcopy(packages_dict)

    mismatches: List[Dict[str, Any]] = []

    # If sales_df is empty or None, return packages with 0 stats
    if sales_df is None or sales_df.empty:
        for cat_key in CATEGORY_ORDER:
            if cat_key in pkgs:
                cat = pkgs[cat_key]
                for p in cat.get("packages", []):
                    p["total_vn"] = 0
                    p["total_revenue"] = 0.0
                cat["total_vn"] = 0
                cat["total_revenue"] = 0.0
        return pkgs, [{"reason": "Sales dataframe is empty or unavailable"}]

    # Normalize sales columns
    sdf = sales_df.copy()
    col_name_map = {}
    for c in sdf.columns:
        c_str = str(c).strip().lower()
        if c_str in ["package_name", "group request"]:
            col_name_map[c] = "package_name"
        elif c_str in ["transactions", "count of vn", "vn", "total_vn"]:
            col_name_map[c] = "transactions"
        elif c_str in ["net_amount", "sum of สุทธิ", "revenue", "sum of revenue", "total_revenue"]:
            col_name_map[c] = "net_amount"

    sdf = sdf.rename(columns=col_name_map)

    if "package_name" not in sdf.columns:
        return pkgs, [{"reason": "Missing package_name column in sales data"}]

    if "transactions" not in sdf.columns:
        sdf["transactions"] = 1
    if "net_amount" not in sdf.columns:
        sdf["net_amount"] = 0.0

    sdf["package_name"] = sdf["package_name"].astype(str).str.strip()
    sdf = sdf[~sdf["package_name"].apply(is_excluded_package)].copy()
    sdf["transactions"] = pd.to_numeric(sdf["transactions"], errors="coerce").fillna(0).astype(int)
    sdf["net_amount"] = pd.to_numeric(sdf["net_amount"], errors="coerce").fillna(0.0).astype(float)

    # Group sales
    grouped = sdf.groupby("package_name").agg(
        total_vn=("transactions", "sum"),
        total_revenue=("net_amount", "sum")
    ).reset_index()

    used_sales_indices = set()

    for cat_key in CATEGORY_ORDER:
        cat = pkgs.get(cat_key)
        if not cat:
            continue

        for p in cat.get("packages", []):
            short_name = p.get("short_name", "")
            full_name = p.get("name", "")

            # 1. Exact match on short_name or full_name
            exact_mask = (grouped["package_name"].str.lower() == short_name.lower()) | \
                         (grouped["package_name"].str.lower() == full_name.lower())

            matched_rows = grouped[exact_mask]

            # 2. If no exact match, use alias keyword matching
            if matched_rows.empty:
                aliases = PACKAGE_KEYWORD_ALIASES.get(short_name, [short_name.lower()])
                pattern = "|".join([a.lower() for a in aliases])
                keyword_mask = grouped["package_name"].str.lower().str.contains(pattern, na=False, regex=True)
                matched_rows = grouped[keyword_mask]

            if not matched_rows.empty:
                p["total_vn"] = int(matched_rows["total_vn"].sum())
                p["total_revenue"] = float(matched_rows["total_revenue"].sum())
                used_sales_indices.update(matched_rows.index.tolist())
            else:
                p["total_vn"] = 0
                p["total_revenue"] = 0.0
                mismatches.append({
                    "category": cat_key,
                    "package_name": full_name,
                    "short_name": short_name,
                    "reason": "ไม่พบรายการขายในฐานข้อมูล RAWDATA Excel"
                })

        # Sort packages within category by total_vn descending
        cat["packages"].sort(key=lambda x: (x.get("total_vn", 0), x.get("total_revenue", 0)), reverse=True)

        # Category level aggregation
        cat["total_vn"] = sum(p.get("total_vn", 0) for p in cat["packages"])
        cat["total_revenue"] = sum(p.get("total_revenue", 0.0) for p in cat["packages"])

    return pkgs, mismatches
