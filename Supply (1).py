import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
from collections import defaultdict, Counter
import sys
import os

# ----------------------
# Input file handling (CSV / XLSX)
# ----------------------

def load_input_dataframe() -> pd.DataFrame:
    """
    Load the input dataset as a DataFrame.
    - If a path is passed as first CLI argument, use it.
    - Otherwise, default to 'Dataset (1).csv' in the current folder.
    Supports .csv and .xlsx files.
    """
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "Dataset (1).csv"

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        df_local = pd.read_csv(input_path)
    elif ext in (".xlsx", ".xls"):
        df_local = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Please use .csv or .xlsx")

    return df_local


# Read the data
df = load_input_dataframe()

# Define the stages in the supply chain
stages = ['Trader', 'Collection Point', 'Cargo Hub', 'WareHouse', 'Distribution Center', 'Store']

# Define color scheme
color_scheme = {
    'Trader': '#1f77b4',  # Blue
    'Collection Point': '#ff7f0e',  # Orange
    'Cargo Hub': '#2ca02c',  # Green
    'WareHouse': '#d62728',  # Red
    'Distribution Center': '#9467bd',  # Purple
    'Store': '#8c564b'  # Brown
}

# Get unique IDs for each stage
trader_ids = df['Trader_ID'].unique()
collection_point_ids = df['Collection Point_ID'].unique()
cargo_hub_ids = df['Cargo Hub_ID'].unique()
warehouse_ids = df['WareHouse_ID'].unique()
distribution_center_ids = df['Distribution Center_ID'].unique()
store_ids = df['Store_ID'].unique()

# Create a mapping of node IDs to their indices in the nodes list
node_indices = {}
current_index = 0

# Create nodes for the Sankey diagram
nodes = []
node_colors = []
node_labels = []

# Add Trader nodes
for trader_id in trader_ids:
    nodes.append(trader_id)
    node_colors.append(color_scheme['Trader'])
    node_labels.append(f"{trader_id}")
    node_indices[f"Trader_{trader_id}"] = current_index
    current_index += 1

# Add Collection Point nodes
for cp_id in collection_point_ids:
    nodes.append(cp_id)
    node_colors.append(color_scheme['Collection Point'])
    node_labels.append(f"{cp_id}")
    node_indices[f"CollectionPoint_{cp_id}"] = current_index
    current_index += 1

# Add Cargo Hub nodes
for ch_id in cargo_hub_ids:
    nodes.append(str(ch_id))
    node_colors.append(color_scheme['Cargo Hub'])
    node_labels.append(str(ch_id))
    node_indices[f"CargoHub_{ch_id}"] = current_index
    current_index += 1

# Add Warehouse nodes
for wh_id in warehouse_ids:
    nodes.append(str(wh_id))
    node_colors.append(color_scheme['WareHouse'])
    node_labels.append(str(wh_id))
    node_indices[f"Warehouse_{wh_id}"] = current_index
    current_index += 1

# Add Distribution Center nodes
for dc_id in distribution_center_ids:
    nodes.append(str(dc_id))
    node_colors.append(color_scheme['Distribution Center'])
    node_labels.append(str(dc_id))
    node_indices[f"DistributionCenter_{dc_id}"] = current_index
    current_index += 1

# Add Store nodes
for store_id in store_ids:
    nodes.append(str(store_id))
    node_colors.append(color_scheme['Store'])
    node_labels.append(str(store_id))
    node_indices[f"Store_{store_id}"] = current_index
    current_index += 1


# Create links for the Sankey diagram and a flat list of link records
sources = []
targets = []
link_values = []
link_colors = []
link_labels = []
link_records = []  # for analytics

stage_names = ['Trader→Collection Point', 'Collection Point→Cargo Hub',
               'Cargo Hub→WareHouse', 'WareHouse→Distribution Center',
               'Distribution Center→Store']

def los_to_color(los_value: float) -> str:
    """Map LOS to a red–green color with intensity based on LOS."""
    # Low LOS = green, high LOS = red, with alpha scaled
    if los_value <= 1:
        alpha = 0.4 + 0.1 * los_value
        return f"rgba(0, 180, 0, {alpha:.2f})"
    elif los_value <= 2:
        alpha = 0.6 + 0.1 * (los_value - 1)
        return f"rgba(200, 120, 0, {alpha:.2f})"
    else:
        alpha = 0.7 + 0.1 * min(los_value - 2, 1)
        return f"rgba(220, 0, 0, {alpha:.2f})"

# Add links between specific nodes based on the data
for _, row in df.iterrows():
    trader_node = f"Trader_{row['Trader_ID']}"
    cp_node = f"CollectionPoint_{row['Collection Point_ID']}"
    ch_node = f"CargoHub_{row['Cargo Hub_ID']}"
    wh_node = f"Warehouse_{row['WareHouse_ID']}"
    dc_node = f"DistributionCenter_{row['Distribution Center_ID']}"
    store_node = f"Store_{row['Store_ID']}"

    # Helper to append a single link and record
    def add_link(source_key, target_key, flow_value, los_value, stage_label):
        source_idx = node_indices[source_key]
        target_idx = node_indices[target_key]
        sources.append(source_idx)
        targets.append(target_idx)
        link_values.append(flow_value)

        los_color = los_to_color(los_value)
        link_colors.append(los_color)
        link_labels.append(f"Flow: {flow_value:.2f}, LOS: {los_value:.2f}")

        link_records.append({
            "source_key": source_key,
            "target_key": target_key,
            "source_label": source_key.split("_", 1)[1],
            "target_label": target_key.split("_", 1)[1],
            "stage": stage_label,
            "flow": float(flow_value),
            "los": float(los_value),
            "trader_id": row['Trader_ID'],
            "store_id": row['Store_ID']
        })

    # Trader to Collection Point
    add_link(trader_node, cp_node, row['Int1'], row['LOS1'], stage_names[0])
    # Collection Point to Cargo Hub
    add_link(cp_node, ch_node, row['Int2'], row['LOS2'], stage_names[1])
    # Cargo Hub to Warehouse
    add_link(ch_node, wh_node, row['Int3'], row['LOS3'], stage_names[2])
    # Warehouse to Distribution Center
    add_link(wh_node, dc_node, row['Int4'], row['LOS4'], stage_names[3])
    # Distribution Center to Store
    add_link(dc_node, store_node, row['Int5'], row['LOS5'], stage_names[4])

# ----------------------
# Bottleneck analytics
# ----------------------

# Aggregate LOS and flow per physical link (stage + from + to)
link_key_totals = defaultdict(lambda: {"total_flow": 0.0, "total_los": 0.0, "count": 0})
node_throughput = defaultdict(float)
node_inflow = defaultdict(float)
node_outflow = defaultdict(float)
vendor_store_los = defaultdict(lambda: {"total_los": 0.0, "count": 0})

for rec in link_records:
    key = (rec["stage"], rec["source_label"], rec["target_label"])
    link_key_totals[key]["total_flow"] += rec["flow"]
    link_key_totals[key]["total_los"] += rec["los"]
    link_key_totals[key]["count"] += 1

    node_throughput[rec["source_label"]] += rec["flow"]
    node_throughput[rec["target_label"]] += rec["flow"]

    node_outflow[rec["source_label"]] += rec["flow"]
    node_inflow[rec["target_label"]] += rec["flow"]

    v_key = (rec["trader_id"], rec["store_id"])
    vendor_store_los[v_key]["total_los"] += rec["los"]
    vendor_store_los[v_key]["count"] += 1

all_flows = [v["total_flow"] for v in link_key_totals.values() if v["total_flow"] > 0]
all_los_values = [v["total_los"] / max(v["count"], 1) for v in link_key_totals.values()]

max_flow = max(all_flows) if all_flows else 1.0
max_los = max(all_los_values) if all_los_values else 1.0

bottleneck_links = []
for (stage, s_lbl, t_lbl), vals in link_key_totals.items():
    avg_los = vals["total_los"] / max(vals["count"], 1)
    norm_los = avg_los / max_los if max_los > 0 else 0.0
    norm_flow = vals["total_flow"] / max_flow if max_flow > 0 else 0.0
    inverse_flow = 1.0 - norm_flow  # lower flow → higher inverse
    bottleneck_score = (norm_los * 0.6 + inverse_flow * 0.4) * 100.0

    bottleneck_links.append({
        "stage": stage,
        "from": s_lbl,
        "to": t_lbl,
        "total_flow": vals["total_flow"],
        "avg_los": avg_los,
        "bottleneck_score": bottleneck_score
    })

# Sort bottlenecks by score
bottleneck_links.sort(key=lambda x: x["bottleneck_score"], reverse=True)

# Highest LOS link (by average LOS)
highest_los_link = max(bottleneck_links, key=lambda x: x["avg_los"]) if bottleneck_links else None

# Lowest flow link (by total flow)
lowest_flow_link = min(bottleneck_links, key=lambda x: x["total_flow"]) if bottleneck_links else None

# Highest capacity mismatch (in-out imbalance)
capacity_mismatch = []
all_nodes = set(node_inflow.keys()) | set(node_outflow.keys())
for n in all_nodes:
    inflow = node_inflow.get(n, 0.0)
    outflow = node_outflow.get(n, 0.0)
    mismatch = abs(inflow - outflow)
    capacity_mismatch.append({
        "node": n,
        "inflow": inflow,
        "outflow": outflow,
        "mismatch": mismatch
    })
capacity_mismatch.sort(key=lambda x: x["mismatch"], reverse=True)
highest_capacity_mismatch = capacity_mismatch[0] if capacity_mismatch else None

# Most repeated vendor-to-store delays
vendor_delay_stats = []
for (vendor, store), vals in vendor_store_los.items():
    avg_vs_los = vals["total_los"] / max(vals["count"], 1)
    vendor_delay_stats.append({
        "vendor": vendor,
        "store": store,
        "avg_los": avg_vs_los,
        "occurrences": vals["count"]
    })
vendor_delay_stats.sort(key=lambda x: (x["avg_los"], x["occurrences"]), reverse=True)
most_repeated_vendor_store_delay = vendor_delay_stats[0] if vendor_delay_stats else None

# ----------------------
# Predictive LOS (simple regression-style model)
# Future_LOS = a*Flow + b*Current_LOS + c
# ----------------------

a_coeff = 0.0005
b_coeff = 0.8
c_coeff = 0.1

predicted_los_values = []
vendor_future_risk_counter = Counter()

for rec in link_records:
    flow = rec["flow"]
    current_los = rec["los"]
    future_los = a_coeff * flow + b_coeff * current_los + c_coeff
    predicted_los_values.append(future_los)
    rec["future_los"] = future_los
    vendor_future_risk_counter[rec["trader_id"]] += future_los

expected_los_30_days = float(np.mean(predicted_los_values)) if predicted_los_values else 0.0

if expected_los_30_days < 1.0:
    expected_congestion_level = "Low"
elif expected_los_30_days < 2.0:
    expected_congestion_level = "Medium"
else:
    expected_congestion_level = "High"

vendor_with_future_risk = None
if vendor_future_risk_counter:
    vendor_with_future_risk = vendor_future_risk_counter.most_common(1)[0][0]

# ----------------------
# Max-Flow Min-Cut based critical edges
# ----------------------

# Build graph with capacities aggregated per (source_idx, target_idx)
graph_capacity = defaultdict(float)
for s_idx, t_idx, val in zip(sources, targets, link_values):
    graph_capacity[(s_idx, t_idx)] += float(val)

all_node_indices = set(node_indices.values())
trader_node_indices = {node_indices[f"Trader_{tid}"] for tid in trader_ids}
store_node_indices = {node_indices[f"Store_{sid}"] for sid in store_ids}

super_source = max(all_node_indices) + 1
super_sink = super_source + 1

for t_idx in trader_node_indices:
    # Connect super source to each trader with large capacity
    graph_capacity[(super_source, t_idx)] += 1e9
for s_idx in store_node_indices:
    # Connect each store to super sink
    graph_capacity[(s_idx, super_sink)] += 1e9

def edmonds_karp(capacity_dict, source, sink):
    """Simple Edmonds–Karp max-flow implementation."""
    # Build adjacency list
    adj = defaultdict(list)
    for (u, v) in capacity_dict.keys():
        adj[u].append(v)
        adj[v].append(u)

    # Residual capacities
    residual = defaultdict(float)
    for (u, v), cap in capacity_dict.items():
        residual[(u, v)] = cap
        residual[(v, u)] = 0.0

    max_flow = 0.0
    parent = {}

    while True:
        # BFS to find augmenting path
        queue = [source]
        parent = {source: None}
        while queue and sink not in parent:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in parent and residual[(u, v)] > 1e-9:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break

        # Find bottleneck capacity
        v = sink
        path_flow = float("inf")
        while parent[v] is not None:
            u = parent[v]
            path_flow = min(path_flow, residual[(u, v)])
            v = u

        # Update residual capacities
        v = sink
        while parent[v] is not None:
            u = parent[v]
            residual[(u, v)] -= path_flow
            residual[(v, u)] += path_flow
            v = u

        max_flow += path_flow

    # Find min-cut: nodes reachable from source in residual graph
    visited = set()
    queue = [source]
    visited.add(source)
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited and residual[(u, v)] > 1e-9:
                visited.add(v)
                queue.append(v)

    cut_edges = []
    for (u, v), cap in capacity_dict.items():
        if u in visited and v not in visited and cap > 0:
            cut_edges.append((u, v))

    return max_flow, cut_edges

max_flow_value, cut_edges = edmonds_karp(graph_capacity, super_source, super_sink)

# Map cut edges back to labels and LOS
critical_edges = []
for u, v in cut_edges:
    if u in node_indices.values() and v in node_indices.values():
        # Find one representative record for this edge
        from_label = node_labels[u]
        to_label = node_labels[v]
        # Find matching link_records entry
        matching = [rec for rec in link_records
                    if rec["source_label"] == from_label and rec["target_label"] == to_label]
        if matching:
            sample = matching[0]
            critical_edges.append({
                "from": from_label,
                "to": to_label,
                "stage": sample["stage"],
                "avg_los": float(np.mean([m["los"] for m in matching])),
                "total_flow": sum(m["flow"] for m in matching)
            })

# Suggest alternate edges with lower LOS on the same stage
alternate_paths = []
for edge in critical_edges:
    same_stage_links = [b for b in bottleneck_links if b["stage"] == edge["stage"]]
    # Find best (lowest LOS) alternative that is not the same physical link
    alternatives = [b for b in same_stage_links
                    if not (b["from"] == edge["from"] and b["to"] == edge["to"])]
    if alternatives:
        best_alt = min(alternatives, key=lambda x: x["avg_los"])
        alternate_paths.append({
            "stage": edge["stage"],
            "current_from": edge["from"],
            "current_to": edge["to"],
            "alternate_from": best_alt["from"],
            "alternate_to": best_alt["to"],
            "current_avg_los": edge["avg_los"],
            "alternate_avg_los": best_alt["avg_los"]
        })

# ----------------------
# KPI scorecard metrics
# ----------------------

# End-to-end LOS: average LOS across all 5 stages per vendor–store pair
end_to_end_los_values = []
for (vendor, store), vals in vendor_store_los.items():
    end_to_end_los_values.append(vals["total_los"] / max(vals["count"], 1))
end_to_end_los = float(np.mean(end_to_end_los_values)) if end_to_end_los_values else 0.0

# Stage throughput: total flow per stage using Int1..Int5
stage_throughput = {
    "Trader→Collection Point": float(df["Int1"].sum() if "Int1" in df.columns else 0.0),
    "Collection Point→Cargo Hub": float(df["Int2"].sum() if "Int2" in df.columns else 0.0),
    "Cargo Hub→WareHouse": float(df["Int3"].sum() if "Int3" in df.columns else 0.0),
    "WareHouse→Distribution Center": float(df["Int4"].sum() if "Int4" in df.columns else 0.0),
    "Distribution Center→Store": float(df["Int5"].sum() if "Int5" in df.columns else 0.0),
}

# Vendor performance: average end-to-end LOS by vendor
vendor_performance = {}
for (vendor, store), vals in vendor_store_los.items():
    vendor_performance.setdefault(str(vendor), []).append(vals["total_los"] / max(vals["count"], 1))
vendor_performance = {
    v: float(np.mean(los_list)) for v, los_list in vendor_performance.items()
}

# Efficiency score and health: simple function of LOS and bottleneck severity (0–100)
avg_bottleneck_score = float(np.mean([b["bottleneck_score"] for b in bottleneck_links])) if bottleneck_links else 0.0

raw_efficiency = max(0.0, 100.0 - (end_to_end_los * 10.0) - (avg_bottleneck_score * 0.2))
efficiency_score = float(min(100.0, raw_efficiency))
total_supply_chain_health = efficiency_score

# Collect analytics for the dashboard
analytics = {
    "kpis": {
        "end_to_end_los": end_to_end_los,
        "stage_throughput": stage_throughput,
        "vendor_performance": vendor_performance,
        "efficiency_score": efficiency_score,
        "total_supply_chain_health": total_supply_chain_health,
    },
    "bottlenecks": {
        "top_bottlenecks": bottleneck_links[:10],
        "highest_los_link": highest_los_link,
        "lowest_flow_link": lowest_flow_link,
        "highest_capacity_mismatch": highest_capacity_mismatch,
        "most_repeated_vendor_store_delay": most_repeated_vendor_store_delay,
        "max_flow": max_flow_value,
        "critical_edges": critical_edges,
        "alternate_paths": alternate_paths,
    },
    "predictive": {
        "expected_los_30_days": expected_los_30_days,
        "expected_congestion": expected_congestion_level,
        "vendor_with_future_risk": str(vendor_with_future_risk) if vendor_with_future_risk is not None else None,
        "coefficients": {"a": a_coeff, "b": b_coeff, "c": c_coeff},
    }
}

# Define fixed positions for each stage - adjusted to ensure all nodes are visible
stage_x = {
    'Trader': 0.02,
    'Collection Point': 0.20,
    'Cargo Hub': 0.46,
    'WareHouse': 0.68,
    'Distribution Center': 0.90,
    'Store': 0.98
}

# Create x and y position arrays
x_positions = []
y_positions = []

# Add positions for each node
for i, node in enumerate(node_labels):
    # Find the stage this node belongs to
    if i < len(trader_ids):
        # Trader nodes
        x_positions.append(stage_x['Trader'])
        if node == 'Vendor1':
            y_positions.append(0.1)
        elif node == 'Vendor2':
            y_positions.append(0.35)
        elif node == 'Vendor3':
            y_positions.append(0.6)
        elif node == 'Vendor4':
            y_positions.append(0.85)
        else:
            y_positions.append(0.5)  # Default
    
    elif i < len(trader_ids) + len(collection_point_ids):
        # Collection Point nodes
        x_positions.append(stage_x['Collection Point'])
        if node == 'c1':
            y_positions.append(0.1)
        elif node == 'c2':
            y_positions.append(0.3)
        elif node == 'c3':
            y_positions.append(0.5)
        elif node == 'c4':
            y_positions.append(0.7)
        elif node == 'c5':
            y_positions.append(0.9)
        else:
            y_positions.append(0.5)  # Default
    
    elif i < len(trader_ids) + len(collection_point_ids) + len(cargo_hub_ids):
        # Cargo Hub nodes
        x_positions.append(stage_x['Cargo Hub'])
        if node == '1':
            y_positions.append(0.1)
        elif node == '2':
            y_positions.append(0.35)
        elif node == '3':
            y_positions.append(0.6)
        elif node == '4':
            y_positions.append(0.85)
        else:
            y_positions.append(0.5)  # Default
    
    elif i < len(trader_ids) + len(collection_point_ids) + len(cargo_hub_ids) + len(warehouse_ids):
        # Warehouse nodes
        x_positions.append(stage_x['WareHouse'])
        if node == '1':
            y_positions.append(0.1)
        elif node == '2':
            y_positions.append(0.35)
        elif node == '3':
            y_positions.append(0.6)
        elif node == '4':
            y_positions.append(0.85)
        else:
            y_positions.append(0.5)  # Default
    
    elif i < len(trader_ids) + len(collection_point_ids) + len(cargo_hub_ids) + len(warehouse_ids) + len(distribution_center_ids):
        # Distribution Center nodes
        x_positions.append(stage_x['Distribution Center'])
        if node == '1':
            y_positions.append(0.1)
        elif node == '2':
            y_positions.append(0.35)
        elif node == '3':
            y_positions.append(0.6)
        elif node == '4':
            y_positions.append(0.85)
        else:
            y_positions.append(0.5)  # Default
    
    else:
        # Store nodes
        x_positions.append(stage_x['Store'])
        if node == '1':
            y_positions.append(0.1)
        elif node == '2':
            y_positions.append(0.35)
        elif node == '3':
            y_positions.append(0.6)
        elif node == '4':
            y_positions.append(0.85)
        else:
            y_positions.append(0.5)  # Default

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels,
        color=node_colors,
        x=x_positions,
        y=y_positions,
        hovertemplate='%{label}<br>Total Flow: %{value}<extra></extra>'
    ),
    link=dict(
        source=sources,
        target=targets,
        value=link_values,
        color=link_colors,
        label=link_labels,
        hovertemplate='%{label}<br>From: %{source.label}<br>To: %{target.label}<extra></extra>'
    )
)])

# Add annotations for the stage names
annotations = []
for stage, x_pos in stage_x.items():
    # Add stage name at the top
    annotations.append(dict(
        x=x_pos,
        y=1.05,
        xref="paper",
        yref="paper",
        text=stage,
        showarrow=False,
        font=dict(size=14, color="black", family="Arial, sans-serif")
    ))

# Update the layout (let Plotly autosize; height will be adapted in JS)
fig.update_layout(
    title_text="Supply Chain Flow Analysis",
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="black"
    ),
    autosize=True,
    margin=dict(l=25, r=25, t=100, b=25),
    annotations=annotations,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Create a list of rows for the data table
table_data = []
for _, row in df.iterrows():
    table_row = [
        row['Trader_ID'], row['Trader'], row['Int1'], row['LOS1'],
        row['Collection Point_ID'], row['Collection Point'], row['Int2'], row['LOS2'],
        row['Cargo Hub_ID'], row['Cargo Hub'], row['Int3'], row['LOS3'],
        row['WareHouse_ID'], row['WareHouse'], row['Int4'], row['LOS4'],
        row['Distribution Center_ID'], row['Distribution Center'], row['Int5'], row['LOS5'],
        row['Store_ID'], row['Store']
    ]
    table_data.append(table_row)

# Convert the table data to a JSON string
table_json = json.dumps(table_data)

# Create HTML with additional styling and interactivity
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Supply Chain Bottleneck Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            min-height: 100vh;
            box-sizing: border-box;
        }
        .header {
            background: linear-gradient(90deg, #0d6efd, #6610f2);
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-title {
            font-size: 22px;
            font-weight: bold;
        }
        .header-subtitle {
            font-size: 13px;
            opacity: 0.9;
        }
        .container {
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 10px;
            box-sizing: border-box;
        }
        .layout {
            display: flex;
            flex-direction: column;
            gap: 16px;
            align-items: stretch;
        }
        .card {
            background-color: white;
            padding: 16px 18px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 14px;
        }
        .card-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        .chart-container {
            width: 100%;
            height: 65vh;
            max-height: 640px;
            min-height: 360px;
            margin-bottom: 4px;
            overflow: visible;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }
        .kpi {
            padding: 10px 12px;
            border-radius: 8px;
            background-color: #f8f9fa;
        }
        .kpi-label {
            font-size: 11px;
            color: #6c757d;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 18px;
            font-weight: 600;
            color: #0d6efd;
        }
        .kpi-sub {
            font-size: 11px;
            color: #6c757d;
        }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            font-size: 11px;
            margin-top: 4px;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 1px solid rgba(0,0,0,0.3);
        }
        .legend-title {
            font-weight: 600;
        }
        .filters {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 4px;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .filter-label {
            font-size: 11px;
            color: #555;
            display: flex;
            justify-content: space-between;
        }
        .filter-value {
            font-weight: 600;
            color: #0d6efd;
            margin-left: 6px;
        }
        .slider {
            width: 100%;
        }
        select, input[type="search"] {
            padding: 5px 8px;
            border-radius: 6px;
            border: 1px solid #ced4da;
            font-size: 12px;
        }
        .toggle-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            font-size: 12px;
        }
        .toggle-row label {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 500;
        }
        .pill-low { background-color: #d1e7dd; color: #0f5132; }
        .pill-medium { background-color: #fff3cd; color: #664d03; }
        .pill-high { background-color: #f8d7da; color: #842029; }
        .bottleneck-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            margin-top: 8px;
        }
        .bottleneck-table th,
        .bottleneck-table td {
            border: 1px solid #dee2e6;
            padding: 6px 8px;
        }
        .bottleneck-table th {
            background: linear-gradient(90deg, #003b73, #0059b3);
            color: #ffffff;
            font-weight: 700;
            font-size: 12px;
            text-align: center;
            white-space: nowrap;
        }
        .bottleneck-table td {
            text-align: center;
        }
        .bottleneck-table td:first-child,
        .bottleneck-table th:first-child {
            text-align: left;
        }
        .bottleneck-row-strong {
            background-color: #fff5f5;
        }
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            align-items: stretch;
        }
        .AT {
            width: 100%;
            margin: 24px auto 0 auto;
            padding: 0 10px 16px 10px;
            box-sizing: border-box;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
            font-size: 11px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 6px;
            text-align: center;
        }
        th {
            background-color: #000;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        /* Responsive layout for smaller screens */
        @media (max-width: 1024px) {
            .chart-container {
                height: 60vh;
                max-height: 520px;
            }
        }

        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 6px;
            }
            .chart-container {
                height: 55vh;
                max-height: 480px;
                min-height: 320px;
            }
            .kpi-grid,
            .kpi-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="header-title">Intelligent Supply Chain Bottleneck Dashboard</div>
                <div class="header-subtitle">
                    Descriptive · Diagnostic · Predictive · Prescriptive analytics across full vendor → store network
                </div>
            </div>
            <div>
                <span id="healthPill" class="pill pill-low">
                    Health: <span id="healthScoreText">–</span>/100
                </span>
            </div>
        </div>

        <div class="layout">
            <!-- Full-width: Sankey + filters -->
            <div class="card">
                <div class="card-title">Flow Visualization & Controls</div>
                <div class="filters">
                    <div class="filter-group">
                        <div class="filter-label">
                            <span>Filter by Vendor</span>
                            <span class="filter-value" id="selectedVendorText">All</span>
                        </div>
                        <select id="vendorFilter">
                            <option value="ALL">All Vendors</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <div class="filter-label">
                            <span>Search Node / Facility</span>
                        </div>
                        <input type="search" id="nodeSearch" placeholder="e.g. Vendor1, c3, 2, Store4">
                    </div>
                </div>
                <div class="filters">
                    <div class="filter-group">
                        <div class="filter-label">
                            <span>Min Link Flow</span>
                            <span class="filter-value" id="linkValueDisplay">>&nbsp;0</span>
                        </div>
                        <input type="range" id="linkValueSlider" min="0" max="1000" value="0" class="slider">
                    </div>
                    <div class="filter-group">
                        <div class="filter-label">
                            <span>Min LOS (Days)</span>
                            <span class="filter-value" id="losValueDisplay">>&nbsp;0</span>
                        </div>
                        <input type="range" id="losSlider" min="0" max="5" value="0" step="0.5" class="slider">
                    </div>
                </div>
                <div class="toggle-row">
                    <label>
                        <input type="checkbox" id="bottleneckToggle">
                        Show only bottleneck links (Top 10)
                    </label>
                    <label>
                        <input type="checkbox" id="highlightPathToggle">
                        Highlight vendor → store critical path
                    </label>
                </div>
                <div class="chart-container" id="sankey-chart"></div>
                <div class="legend">
                    <span class="legend-title">Node Colors:</span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:#1f77b4;"></span> Trader
                    </span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:#ff7f0e;"></span> Collection Point
                    </span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:#2ca02c;"></span> Cargo Hub
                    </span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:#d62728;"></span> Warehouse
                    </span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:#9467bd;"></span> Distribution Center
                    </span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:#8c564b;"></span> Store
                    </span>
                </div>
                <div class="legend" style="margin-top:6px;">
                    <span class="legend-title">Link LOS Intensity:</span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:rgba(0, 180, 0, 0.5);"></span> Low LOS
                    </span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:rgba(200, 120, 0, 0.7);"></span> Medium LOS
                    </span>
                    <span class="legend-item">
                        <span class="legend-color" style="background-color:rgba(220, 0, 0, 0.8);"></span> High LOS
                    </span>
                </div>
            </div>

            <!-- Second row: KPIs, Bottlenecks, Predictive in three columns -->
            <div class="kpi-row">
                <div class="card">
                    <div class="card-title">Scorecard KPIs</div>
                    <div class="kpi-grid">
                        <div class="kpi">
                            <div class="kpi-label">End-to-End LOS (Avg Days)</div>
                            <div class="kpi-value" id="kpiEndToEndLos">–</div>
                        </div>
                        <div class="kpi">
                            <div class="kpi-label">Network Throughput</div>
                            <div class="kpi-value" id="kpiThroughput">–</div>
                            <div class="kpi-sub" id="kpiThroughputDetail"></div>
                        </div>
                        <div class="kpi">
                            <div class="kpi-label">Efficiency Score</div>
                            <div class="kpi-value" id="kpiEfficiency">–</div>
                            <div class="kpi-sub">Higher is better (0–100)</div>
                        </div>
                    </div>
                    <div style="margin-top:10px; font-size:11px;">
                        <strong>Vendor Performance (Avg LOS):</strong>
                        <span id="vendorPerformanceText"></span>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">Top Bottlenecks & Critical Paths</div>
                    <table class="bottleneck-table" id="bottleneckTable">
                        <thead>
                            <tr>
                                <th>Stage</th>
                                <th>From → To</th>
                                <th>Flow</th>
                                <th>LOS</th>
                                <th>Bottleneck Score</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                    <div style="margin-top:8px; font-size:11px;" id="criticalPathSummary"></div>
                </div>

                <div class="card">
                    <div class="card-title">Predictive & Optimization Insights (30 Days)</div>
                    <div style="font-size:11px; margin-bottom:6px;">
                        <strong>Expected LOS:</strong>
                        <span id="predictedLosText">–</span>
                        <span id="predictedLosPill" class="pill pill-low" style="margin-left:6px;">–</span>
                    </div>
                    <div style="font-size:11px; margin-bottom:4px;">
                        <strong>Vendor with Future Risk:</strong>
                        <span id="futureRiskVendor">–</span>
                    </div>
                    <div style="font-size:11px;">
                        <strong>Alternate Route Suggestions:</strong>
                        <ul id="alternateRouteList" style="padding-left:16px; margin:4px 0;"></ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="AT">
        <h3>Supply Chain Dataset</h3>
        <table id="dataTable">
            <thead>
                <tr>
                    <th>Trader_ID</th>
                    <th>Trader</th>
                    <th>Int1</th>
                    <th>LOS1</th>
                    <th>Collection Point_ID</th>
                    <th>Collection Point</th>
                    <th>Int2</th>
                    <th>LOS2</th>
                    <th>Cargo Hub_ID</th>
                    <th>Cargo Hub</th>
                    <th>Int3</th>
                    <th>LOS3</th>
                    <th>WareHouse_ID</th>
                    <th>WareHouse</th>
                    <th>Int4</th>
                    <th>LOS4</th>
                    <th>Distribution Center_ID</th>
                    <th>Distribution Center</th>
                    <th>Int5</th>
                    <th>LOS5</th>
                    <th>Store_ID</th>
                    <th>Store</th>
                </tr>
            </thead>
            <tbody id="tableBody">
                <!-- Table data will be populated by JavaScript -->
            </tbody>
        </table>
    </div>
    
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <script>
        // Insert the Plotly figure and analytics
        const sankeyData = PLOTLY_FIGURE;
        const tableData = TABLE_DATA;
        const analytics = ANALYTICS_DATA;
        
        // Helper to size the chart to current viewport/container
        function sizeSankeyLayout(layout) {
            const container = document.getElementById('sankey-chart');
            const rect = container.getBoundingClientRect();
            const viewportH = window.innerHeight || 700;
            const targetHeight = Math.min(Math.max(viewportH * 0.6, 320), 700);
            layout.width = rect.width;
            layout.height = targetHeight;
        }

        // Create the plot with responsive layout
        sizeSankeyLayout(sankeyData.layout);
        Plotly.newPlot('sankey-chart', sankeyData.data, sankeyData.layout, {
            responsive: true,
            displayModeBar: false
        });

        // Ensure the chart is fully visible and resizes with window
        function relayoutSankey() {
            const newLayout = Object.assign({}, sankeyData.layout);
            sizeSankeyLayout(newLayout);
            Plotly.relayout('sankey-chart', {
                width: newLayout.width,
                height: newLayout.height
            });
        }

        window.addEventListener('load', function() {
            setTimeout(relayoutSankey, 400);
        });
        window.addEventListener('resize', function() {
            relayoutSankey();
        });
        
        // Populate the data table
        const tableBody = document.getElementById('tableBody');
        tableData.forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell;
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });

        // Populate vendor filter
        const vendorFilter = document.getElementById('vendorFilter');
        const selectedVendorText = document.getElementById('selectedVendorText');
        const vendorSet = new Set(tableData.map(r => r[0]));
        vendorSet.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v;
            opt.textContent = v;
            vendorFilter.appendChild(opt);
        });

        // KPIs
        function formatNumber(x) {
            return x.toLocaleString(undefined, { maximumFractionDigits: 1 });
        }

        const kpiEndToEndLos = document.getElementById('kpiEndToEndLos');
        const kpiThroughput = document.getElementById('kpiThroughput');
        const kpiThroughputDetail = document.getElementById('kpiThroughputDetail');
        const kpiEfficiency = document.getElementById('kpiEfficiency');
        const vendorPerformanceText = document.getElementById('vendorPerformanceText');
        const healthScoreText = document.getElementById('healthScoreText');
        const healthPill = document.getElementById('healthPill');

        const kpis = analytics.kpis;
        kpiEndToEndLos.textContent = kpis.end_to_end_los.toFixed(2);

        const stageThroughput = kpis.stage_throughput;
        const totalThroughput = Object.values(stageThroughput).reduce((a, b) => a + b, 0);
        kpiThroughput.textContent = formatNumber(totalThroughput);
        kpiThroughputDetail.textContent = Object.entries(stageThroughput)
            .map(([stage, val]) => stage.split('→')[0] + ': ' + Math.round(val))
            .join(' · ');

        kpiEfficiency.textContent = kpis.efficiency_score.toFixed(1);
        healthScoreText.textContent = kpis.total_supply_chain_health.toFixed(0);

        // Health pill coloring
        const health = kpis.total_supply_chain_health;
        healthPill.classList.remove('pill-low', 'pill-medium', 'pill-high');
        if (health >= 75) {
            healthPill.classList.add('pill-low');
        } else if (health >= 50) {
            healthPill.classList.add('pill-medium');
        } else {
            healthPill.classList.add('pill-high');
        }

        // Vendor performance text
        const vp = kpis.vendor_performance;
        const vpParts = Object.entries(vp).map(([vendor, los]) => `${vendor}: ${los.toFixed(2)}d`);
        vendorPerformanceText.textContent = vpParts.join(' · ');

        // Bottleneck table
        const bottleneckTableBody = document.querySelector('#bottleneckTable tbody');
        const topBottlenecks = analytics.bottlenecks.top_bottlenecks || [];
        topBottlenecks.forEach((b, idx) => {
            const tr = document.createElement('tr');
            if (idx === 0) {
                tr.classList.add('bottleneck-row-strong');
            }
            tr.innerHTML = `
                <td>${b.stage}</td>
                <td>${b.from} → ${b.to}</td>
                <td>${Math.round(b.total_flow)}</td>
                <td>${b.avg_los.toFixed(2)}</td>
                <td>${b.bottleneck_score.toFixed(1)}</td>
            `;
            bottleneckTableBody.appendChild(tr);
        });

        // Critical path + alternates
        const crit = analytics.bottlenecks;
        const criticalPathSummary = document.getElementById('criticalPathSummary');
        if ((crit.critical_edges || []).length > 0) {
            const edges = crit.critical_edges
                .map(e => `${e.stage}: ${e.from} → ${e.to}`)
                .join(' · ');
            criticalPathSummary.textContent =
                `Critical cut-set edges (Max-Flow / Min-Cut): ${edges}`;
        } else {
            criticalPathSummary.textContent = 'No critical cut-set edges identified.';
        }

        // Predictive info
        const predictedLosText = document.getElementById('predictedLosText');
        const predictedLosPill = document.getElementById('predictedLosPill');
        const futureRiskVendor = document.getElementById('futureRiskVendor');
        const alternateRouteList = document.getElementById('alternateRouteList');

        const pred = analytics.predictive;
        predictedLosText.textContent = `${pred.expected_los_30_days.toFixed(2)} days`;
        predictedLosPill.textContent = pred.expected_congestion + ' congestion';
        predictedLosPill.classList.remove('pill-low', 'pill-medium', 'pill-high');
        if (pred.expected_congestion === 'Low') {
            predictedLosPill.classList.add('pill-low');
        } else if (pred.expected_congestion === 'Medium') {
            predictedLosPill.classList.add('pill-medium');
        } else {
            predictedLosPill.classList.add('pill-high');
        }
        futureRiskVendor.textContent = pred.vendor_with_future_risk || '–';

        (analytics.bottlenecks.alternate_paths || []).forEach(alt => {
            const li = document.createElement('li');
            li.textContent =
                `${alt.stage}: reroute from ${alt.current_from}→${alt.current_to} ` +
                `to ${alt.alternate_from}→${alt.alternate_to} ` +
                `(LOS ${alt.current_avg_los.toFixed(2)} → ${alt.alternate_avg_los.toFixed(2)})`;
            alternateRouteList.appendChild(li);
        });

        // Interactive filters: vendor, LOS, flow, bottlenecks
        const linkValueSlider = document.getElementById('linkValueSlider');
        const losSlider = document.getElementById('losSlider');
        const linkValueDisplay = document.getElementById('linkValueDisplay');
        const losValueDisplay = document.getElementById('losValueDisplay');
        const bottleneckToggle = document.getElementById('bottleneckToggle');
        const nodeSearch = document.getElementById('nodeSearch');
        const highlightPathToggle = document.getElementById('highlightPathToggle');

        const originalSankey = JSON.parse(JSON.stringify(sankeyData));

        function applyFilters() {
            const minFlow = parseFloat(linkValueSlider.value);
            const minLos = parseFloat(losSlider.value);
            const vendor = vendorFilter.value;
            const showOnlyBottlenecks = bottleneckToggle.checked;
            const rawSearch = nodeSearch.value.trim().toLowerCase();
            const searchTerms = rawSearch
                .split(',')
                .map(t => t.trim())
                .filter(t => t.length > 0);

            linkValueDisplay.textContent = '>' + Math.round(minFlow);
            losValueDisplay.textContent = '>' + minLos;
            selectedVendorText.textContent = vendor === 'ALL' ? 'All' : vendor;

            const dataCopy = JSON.parse(JSON.stringify(originalSankey));
            const link = dataCopy.data[0].link;
            const node = dataCopy.data[0].node;

            const newSource = [];
            const newTarget = [];
            const newValue = [];
            const newColor = [];
            const newLabel = [];

            // Parse vendor info from labels (we know trader labels start with "Vendor")
            const traderIndices = [];
            node.label.forEach((lbl, idx) => {
                if (lbl && lbl.toString().toLowerCase().startsWith('vendor')) {
                    traderIndices.push({ idx, vendor: lbl });
                }
            });

            function linkBelongsToVendor(sourceIdx) {
                if (vendor === 'ALL') return true;
                const match = traderIndices.find(t => t.idx === sourceIdx);
                return match ? (match.vendor === vendor) : true;
            }

            for (let i = 0; i < link.value.length; i++) {
                const val = link.value[i];
                const lbl = link.label[i] || '';
                const losMatch = lbl.match(/LOS:\s*([0-9.]+)/);
                const los = losMatch ? parseFloat(losMatch[1]) : 0;
                const src = link.source[i];
                const tgt = link.target[i];

                if (val < minFlow) continue;
                if (los < minLos) continue;
                if (!linkBelongsToVendor(src)) continue;

                // If there is a node search, keep only links touching any searched node
                if (searchTerms.length > 0) {
                    const fromLabel = node.label[src] ? node.label[src].toString().toLowerCase() : '';
                    const toLabel = node.label[tgt] ? node.label[tgt].toString().toLowerCase() : '';
                    const matchesSearch = searchTerms.some(term =>
                        fromLabel.includes(term) || toLabel.includes(term)
                    );
                    if (!matchesSearch) continue;
                }

                // Bottleneck-only mode: keep only links present in top bottlenecks
                if (showOnlyBottlenecks) {
                    const fromLabel = node.label[src];
                    const toLabel = node.label[tgt];
                    const isBottleneck = topBottlenecks.some(b => b.from === fromLabel && b.to === toLabel);
                    if (!isBottleneck) continue;
                }

                newSource.push(src);
                newTarget.push(tgt);
                newValue.push(val);
                newColor.push(link.color[i]);
                newLabel.push(link.label[i]);
            }

            link.source = newSource;
            link.target = newTarget;
            link.value = newValue;
            link.color = newColor;
            link.label = newLabel;

            // Node highlight on search
            const baseNodeColors = originalSankey.data[0].node.color;
            const newNodeColors = baseNodeColors.slice();
            if (searchTerms.length > 0) {
                node.label.forEach((lbl, idx) => {
                    const lower = lbl ? lbl.toString().toLowerCase() : '';
                    const hit = searchTerms.some(term => lower.includes(term));
                    if (hit) {
                        newNodeColors[idx] = '#ffc107';
                    }
                });
            }
            node.color = newNodeColors;

            Plotly.react('sankey-chart', dataCopy.data, dataCopy.layout, {
                responsive: true,
                displayModeBar: false
            });
        }

        linkValueSlider.addEventListener('input', applyFilters);
        losSlider.addEventListener('input', applyFilters);
        bottleneckToggle.addEventListener('change', applyFilters);
        vendorFilter.addEventListener('change', applyFilters);
        nodeSearch.addEventListener('input', applyFilters);

        // Optional: highlight critical path edges by thickening them
        highlightPathToggle.addEventListener('change', function() {
            const checked = highlightPathToggle.checked;
            const dataCopy = JSON.parse(JSON.stringify(originalSankey));
            const criticalEdges = analytics.bottlenecks.critical_edges || [];

            if (checked && criticalEdges.length > 0) {
                const link = dataCopy.data[0].link;
                const node = dataCopy.data[0].node;
                const widths = new Array(link.value.length).fill(8);

                for (let i = 0; i < link.value.length; i++) {
                    const fromLabel = node.label[link.source[i]];
                    const toLabel = node.label[link.target[i]];
                    const isCritical = criticalEdges.some(
                        e => e.from === fromLabel && e.to === toLabel
                    );
                    if (isCritical) {
                        widths[i] = 20;
                    }
                }
                dataCopy.data[0].link.line = { color: 'rgba(0,0,0,0.5)', width: widths };
            }

            Plotly.react('sankey-chart', dataCopy.data, dataCopy.layout, {
                responsive: true,
                displayModeBar: false
            });
        });

        // Initialize filters
        applyFilters();
    </script>
</body>
</html>
"""

# Save the figure and analytics as JSON
plotly_figure_json = json.dumps(fig.to_dict())
analytics_json = json.dumps(analytics)

# Replace the placeholders in the HTML content
html_content = html_content.replace('PLOTLY_FIGURE', plotly_figure_json)
html_content = html_content.replace('TABLE_DATA', table_json)
html_content = html_content.replace('ANALYTICS_DATA', analytics_json)

# Save the HTML file
with open('sankey_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Sankey dashboard with bottleneck analytics created successfully. Open sankey_dashboard.html to view it.")