import json
import yaml
from collections import defaultdict, deque
import torch
import time
import os

# لود تنظیمات از فایل YAML
config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# استخراج پارامترهای تنظیمات
num_epochs = config["model"]["num_epochs"]
num_samples = config["model"]["num_samples"]
num_layers = config["model"]["num_layers"]
gpu_hidden_dim = config["model"]["gpu_hidden_dim"]
cpu_hidden_dim = config["model"]["cpu_hidden_dim"]
device = config["model"]["device"]
charnum_s = config["model"]["charnum_s"]
charnum_n = config["model"]["charnum_n"]
charnum_se = config["model"]["charnum_se"]
charnum_ne = config["model"]["charnum_ne"]
charnum_node = config["model"]["charnum_node"]
charnum_component = config["model"]["charnum_component"]

# تعیین دستگاه و dim
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        hidden_dim = gpu_hidden_dim
    else:
        device = "cpu"
        hidden_dim = cpu_hidden_dim
else:
    device = device.lower()
    hidden_dim = gpu_hidden_dim if device == "cuda" else cpu_hidden_dim

    # لود یا مقداردهی اولیه پارامترها
    # parameters = initialize_settings(
    # config_path = "configs/config.yaml"


# شروع زمان‌سنجی
total_start_time = time.time()

# تعریف دایرکتوری خروجی
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)


def make_undirected_matrix(matrix):
    # تبدیل ماتریس جهت‌دار به دوطرفه
    n = len(matrix)
    undirected = [row[:] for row in matrix]  # کپی ماتریس
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1 or matrix[j][i] == 1:
                undirected[i][j] = 1
                undirected[j][i] = 1
    return undirected


def calculate_distance(graph, start, end):
    # محاسبه فاصله بین دو گره با BFS
    if start == end:
        return 0
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)

    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                if neighbor == end:
                    return dist + 1
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float("inf")  # اگر مسیری وجود نداشته باشه


component = []


def assign_weights(services_edge, charnum_component):
    # تبدیل ماتریس به گراف
    graph = defaultdict(list)
    undirected_matrix = make_undirected_matrix(services_edge)
    for i in range(charnum_component):
        for j in range(charnum_component):
            if undirected_matrix[i][j] == 1:
                graph[f"c{i+1}"].append(f"c{j+1}")

    # تخصیص وزن برای همه جفت‌های ممکن
    weighted_edges = []
    components = [f"c{i+1}" for i in range(charnum_component)]
    for i, c1 in enumerate(components):
        for c2 in components[i:]:  # برای جلوگیری از تکرار
            distance = calculate_distance(graph, c1, c2)
            if distance == 0:
                weight = 1.0
            elif distance == 1:
                weight = 0.75
            elif distance == 2:
                weight = 0.5
            elif distance == 3:
                weight = 0.25
            else:
                weight = 0.1
            if c1 != c2:  # خودگره رو نادیده بگیر
                weighted_edges.append([c1, c2, weight])
                weighted_edges.append([c2, c1, weight])  # مسیر معکوس

    return weighted_edges


def process_services(json_path, charnum_component):
    with open(json_path, "r") as f:
        data = json.load(f)

    services_edge = data.get("componentConnections", [])
    if not services_edge:
        print("No component connections found.")
        return []

    weighted_edges = assign_weights(services_edge, charnum_component)
    return weighted_edges


# لیست برای ذخیره نتایج
results = []


def extract_submatrix(nodes_edge, charnum_node):
    """
    استخراج زیرماتریس charnum_node x charnum_node از گوشه‌ی بالا-چپ ماتریس nodes_edge.

    Args:
        nodes_edge (list): ماتریس شبکه (لیست دوبعدی)
        charnum_node (int): اندازه زیرماتریس (تعداد گره‌ها)

    Returns:
        list: زیرماتریس charnum_node x charnum_node
    """
    try:
        # 1. بررسی اینکه nodes_edge یک لیست است
        if not isinstance(nodes_edge, list) or not nodes_edge:
            raise ValueError("nodes_edge باید یک لیست غیرخالی باشد!")

        # 2. بررسی اندازه ماتریس
        if not isinstance(charnum_node, int) or charnum_node <= 0:
            raise ValueError("مقدار charnum_node باید یک عدد صحیح مثبت باشد!")
        if charnum_node > len(nodes_edge) or charnum_node > len(nodes_edge[0]):
            raise ValueError(
                f"مقدار charnum_node ({charnum_node}) بزرگ‌تر از اندازه ماتریس ({len(nodes_edge)}) است!"
            )

        # 3. استخراج زیرماتریس
        submatrix = []
        for i in range(charnum_node):
            row = nodes_edge[i][:charnum_node]  # انتخاب ستون‌های 0 تا charnum_node-1
            submatrix.append(row)

        # 4. ذخیره زیرماتریس در فایل جدید (اختیاری)
        output_path = os.path.join(output_dir, f"nodes_edge_submatrix.json")
        with open(output_path, "w") as f:
            json.dump(submatrix, f, indent=4)

        return submatrix

    except Exception as e:
        print(f"خطا در extract_submatrix: {e}")
        return None


# حلقه برای پردازش instanceها
for x in range(1, num_samples + 1):  # برای همه 64 instance
    file_path = f"data/generated/small/{x}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            print(f"Processing {x}.json")
            # استخراج بخش‌ها
            component = data.get("componentConnections", [])
            sections = {
                "nodes": data.get("computingNodes", []),
                "helpers": data.get("helperNodes", []),
                "users": data.get("usersNodes", []),
                "services": data.get("services", []),
                "services_edge": data.get("componentConnections", []),
                "nodes_edge": data.get("networkConnections", []),
                "results": data.get("results", {}),
            }
            # وزن‌دهی به componentConnections
            sections["services_edge"] = assign_weights(
                sections["services_edge"], charnum_component
            )
            # networkConnections
            sections["nodes_edge"] = extract_submatrix(
                sections["nodes_edge"], charnum_node
            )

            # ذخیره بخش‌ها در فایل‌های JSON
            for section_name, section_data in sections.items():
                output_file = os.path.join(output_dir, f"{section_name}_{x}.json")
                with open(output_file, "w") as out_f:
                    json.dump(section_data, out_f, indent=4)
    # result = process_services(, "configs/config.yaml")
