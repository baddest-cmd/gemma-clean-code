"""Synthetic AST Corpus Generator for Mechanistic Interpretability.

Constructs contrastive Python snippet pairs (Target construct vs. Near-Miss control)
across 5 key AST categories:
1. try_except    (ast.Try vs. conditional status / guard checks)
2. with_statement (ast.With vs. manual acquire-try-release / explicit close)
3. async_await   (ast.AsyncFunctionDef / ast.Await vs. synchronous equivalents)
4. class_def     (ast.ClassDef vs. closure / dictionary / procedural modules)
5. nested_loop   (nested ast.For/While vs. flat 1D index / itertools product)

Every snippet is strictly parsed and verified via Python's native `ast.parse()`.
Outputs are written to `corpus/gemma_ast_corpus.jsonl`.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Set


class ASTNodeCollector(ast.NodeVisitor):
    """Collects set of AST node class names present in an AST."""

    def __init__(self):
        self.node_names: Set[str] = set()

    def generic_visit(self, node):
        self.node_names.add(type(node).__name__)
        super().generic_visit(node)


def extract_ast_nodes(code: str) -> List[str]:
    """Parse python code and return sorted list of AST node class names."""
    tree = ast.parse(code)
    collector = ASTNodeCollector()
    collector.visit(tree)
    return sorted(list(collector.node_names))


def validate_python_code(code: str, label: str = "code") -> bool:
    """Validate that code parses under Python AST."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        raise ValueError(f"Syntax error in {label}: {e}\nCode:\n{code}") from e


def create_try_except_pairs() -> List[Dict[str, Any]]:
    """Build contrastive pairs for try_except (Target: ast.Try vs Control: guard check)."""
    templates = [
        {
            "desc": "Division by zero handling",
            "target": """def safe_divide(numerator: float, denominator: float) -> float:
    try:
        result = numerator / denominator
        return result
    except ZeroDivisionError:
        return 0.0
""",
            "control": """def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    else:
        result = numerator / denominator
        return result
""",
        },
        {
            "desc": "Dictionary key lookup fallback",
            "target": """def get_config_value(settings: dict, key: str, default: int = -1) -> int:
    try:
        val = settings[key]
        return int(val)
    except KeyError:
        return default
""",
            "control": """def get_config_value(settings: dict, key: str, default: int = -1) -> int:
    if key in settings:
        val = settings[key]
        return int(val)
    else:
        return default
""",
        },
        {
            "desc": "String integer conversion",
            "target": """def parse_port_number(raw_port: str) -> int:
    try:
        port = int(raw_port)
        return port
    except ValueError:
        return 8080
""",
            "control": """def parse_port_number(raw_port: str) -> int:
    if raw_port.isdigit():
        port = int(raw_port)
        return port
    else:
        return 8080
""",
        },
        {
            "desc": "List index access boundary",
            "target": """def get_nth_element(items: list, index: int, fallback: str = "") -> str:
    try:
        element = items[index]
        return str(element)
    except IndexError:
        return fallback
""",
            "control": """def get_nth_element(items: list, index: int, fallback: str = "") -> str:
    if 0 <= index < len(items):
        element = items[index]
        return str(element)
    else:
        return fallback
""",
        },
        {
            "desc": "JSON string parsing guard",
            "target": """import json

def parse_payload(payload_str: str) -> dict:
    try:
        data = json.loads(payload_str)
        return data
    except json.JSONDecodeError:
        return {}
""",
            "control": """import json

def parse_payload(payload_str: str) -> dict:
    if payload_str.startswith("{") and payload_str.endswith("}"):
        data = json.loads(payload_str)
        return data
    else:
        return {}
""",
        },
        {
            "desc": "File opening with error handling",
            "target": """def read_cache_file(filepath: str) -> str:
    try:
        f = open(filepath, "r")
        content = f.read()
        f.close()
        return content
    except FileNotFoundError:
        return ""
""",
            "control": """import os

def read_cache_file(filepath: str) -> str:
    if os.path.exists(filepath):
        f = open(filepath, "r")
        content = f.read()
        f.close()
        return content
    else:
        return ""
""",
        },
        {
            "desc": "Attribute resolution on object",
            "target": """def get_version_info(module_obj: object) -> str:
    try:
        ver = module_obj.__version__
        return str(ver)
    except AttributeError:
        return "unknown"
""",
            "control": """def get_version_info(module_obj: object) -> str:
    if hasattr(module_obj, "__version__"):
        ver = module_obj.__version__
        return str(ver)
    else:
        return "unknown"
""",
        },
        {
            "desc": "Float parsing from scientific string",
            "target": """def extract_learning_rate(token: str) -> float:
    try:
        lr = float(token)
        return lr
    except ValueError:
        return 0.001
""",
            "control": """def extract_learning_rate(token: str) -> float:
    cleaned = token.replace(".", "", 1).replace("e-", "", 1).replace("e+", "", 1)
    if cleaned.isdigit():
        lr = float(token)
        return lr
    else:
        return 0.001
""",
        },
        {
            "desc": "Pop item from stack",
            "target": """def safe_pop_token(stack: list) -> str:
    try:
        item = stack.pop()
        return str(item)
    except IndexError:
        return "<EMPTY>"
""",
            "control": """def safe_pop_token(stack: list) -> str:
    if len(stack) > 0:
        item = stack.pop()
        return str(item)
    else:
        return "<EMPTY>"
""",
        },
        {
            "desc": "Square root of non-negative number",
            "target": """import math

def compute_root(value: float) -> float:
    try:
        res = math.sqrt(value)
        return res
    except ValueError:
        return 0.0
""",
            "control": """import math

def compute_root(value: float) -> float:
    if value >= 0.0:
        res = math.sqrt(value)
        return res
    else:
        return 0.0
""",
        },
    ]

    pairs = []
    for i, t in enumerate(templates, 1):
        pairs.append(
            {
                "pair_id": f"try_except_{i:03d}",
                "category": "try_except",
                "target_construct": "Try",
                "diff_summary": t["desc"],
                "target_code": t["target"],
                "control_code": t["control"],
            }
        )
    return pairs


def create_with_statement_pairs() -> List[Dict[str, Any]]:
    """Build contrastive pairs for with_statement (Target: ast.With vs Control: manual open/close)."""
    templates = [
        {
            "desc": "File reading context manager",
            "target": """def read_lines(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as handle:
        data = handle.readlines()
    return data
""",
            "control": """def read_lines(filepath: str) -> list:
    handle = open(filepath, "r", encoding="utf-8")
    data = handle.readlines()
    handle.close()
    return data
""",
        },
        {
            "desc": "File write context manager",
            "target": """def write_log_entry(filepath: str, entry: str) -> int:
    with open(filepath, "a", encoding="utf-8") as f:
        bytes_written = f.write(entry + "\\n")
    return bytes_written
""",
            "control": """def write_log_entry(filepath: str, entry: str) -> int:
    f = open(filepath, "a", encoding="utf-8")
    bytes_written = f.write(entry + "\\n")
    f.close()
    return bytes_written
""",
        },
        {
            "desc": "Threading Lock acquisition",
            "target": """import threading

def update_counter(lock: threading.Lock, state: dict, delta: int) -> int:
    with lock:
        state["count"] += delta
        current = state["count"]
    return current
""",
            "control": """import threading

def update_counter(lock: threading.Lock, state: dict, delta: int) -> int:
    lock.acquire()
    state["count"] += delta
    current = state["count"]
    lock.release()
    return current
""",
        },
        {
            "desc": "Temporary redirect stdout buffer",
            "target": """import io
from contextlib import redirect_stdout

def capture_output(action) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        action()
    return buf.getvalue()
""",
            "control": """import io
import sys

def capture_output(action) -> str:
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    action()
    sys.stdout = old_stdout
    return buf.getvalue()
""",
        },
        {
            "desc": "Database transaction block",
            "target": """def execute_transaction(session, query: str) -> bool:
    with session.begin():
        res = session.execute(query)
    return res.rowcount > 0
""",
            "control": """def execute_transaction(session, query: str) -> bool:
    tx = session.begin()
    res = session.execute(query)
    tx.commit()
    return res.rowcount > 0
""",
        },
        {
            "desc": "Multiple context managers simultaneously",
            "target": """def copy_transformed(src_path: str, dst_path: str) -> int:
    with open(src_path, "r") as src, open(dst_path, "w") as dst:
        content = src.read().upper()
        count = dst.write(content)
    return count
""",
            "control": """def copy_transformed(src_path: str, dst_path: str) -> int:
    src = open(src_path, "r")
    dst = open(dst_path, "w")
    content = src.read().upper()
    count = dst.write(content)
    src.close()
    dst.close()
    return count
""",
        },
        {
            "desc": "PyTorch inference mode context",
            "target": """import torch

def compute_embeddings(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(inputs)
    return out.logits
""",
            "control": """import torch

def compute_embeddings(model, inputs) -> torch.Tensor:
    prev_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    out = model(inputs)
    torch.set_grad_enabled(prev_state)
    return out.logits
""",
        },
        {
            "desc": "ZipFile archive extraction",
            "target": """import zipfile

def list_archive_members(zip_path: str) -> list:
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
    return names
""",
            "control": """import zipfile

def list_archive_members(zip_path: str) -> list:
    archive = zipfile.ZipFile(zip_path, "r")
    names = archive.namelist()
    archive.close()
    return names
""",
        },
        {
            "desc": "Socket connection context",
            "target": """import socket

def ping_host(host: str, port: int) -> bool:
    with socket.create_connection((host, port), timeout=2.0) as sock:
        sock.sendall(b"PING")
        reply = sock.recv(4)
    return reply == b"PONG"
""",
            "control": """import socket

def ping_host(host: str, port: int) -> bool:
    sock = socket.create_connection((host, port), timeout=2.0)
    sock.sendall(b"PING")
    reply = sock.recv(4)
    sock.close()
    return reply == b"PONG"
""",
        },
        {
            "desc": "Temporary working directory context",
            "target": """import os

class TempDirContext:
    def __init__(self, path): self.path = path
    def __enter__(self): os.chdir(self.path); return self.path
    def __exit__(self, *args): os.chdir("/")

def run_in_dir(path: str, action) -> None:
    with TempDirContext(path):
        action()
""",
            "control": """import os

def run_in_dir(path: str, action) -> None:
    original = os.getcwd()
    os.chdir(path)
    action()
    os.chdir(original)
""",
        },
    ]

    pairs = []
    for i, t in enumerate(templates, 1):
        pairs.append(
            {
                "pair_id": f"with_statement_{i:03d}",
                "category": "with_statement",
                "target_construct": "With",
                "diff_summary": t["desc"],
                "target_code": t["target"],
                "control_code": t["control"],
            }
        )
    return pairs


def create_async_await_pairs() -> List[Dict[str, Any]]:
    """Build contrastive pairs for async_await (Target: AsyncFunctionDef/Await vs Control: sync def)."""
    templates = [
        {
            "desc": "Asynchronous HTTP fetch vs synchronous fetch",
            "target": """import asyncio

async def fetch_endpoint(url: str, client) -> str:
    response = await client.get(url)
    data = await response.text()
    return data
""",
            "control": """def fetch_endpoint(url: str, client) -> str:
    response = client.get(url)
    data = response.text()
    return data
""",
        },
        {
            "desc": "Asynchronous delay sleep vs synchronous sleep",
            "target": """import asyncio

async def poll_status_async(job_id: str, client, max_retries: int) -> bool:
    for _ in range(max_retries):
        await asyncio.sleep(0.5)
        status = await client.check_job(job_id)
        if status == "COMPLETE":
            return True
    return False
""",
            "control": """import time

def poll_status_async(job_id: str, client, max_retries: int) -> bool:
    for _ in range(max_retries):
        time.sleep(0.5)
        status = client.check_job(job_id)
        if status == "COMPLETE":
            return True
    return False
""",
        },
        {
            "desc": "Async generator stream consumption",
            "target": """async def accumulate_stream(stream) -> list:
    results = []
    async for chunk in stream:
        results.append(chunk)
    return results
""",
            "control": """def accumulate_stream(stream) -> list:
    results = []
    for chunk in stream:
        results.append(chunk)
    return results
""",
        },
        {
            "desc": "Async context manager on connection pool",
            "target": """async def acquire_record(pool, record_id: int) -> dict:
    async with pool.acquire() as conn:
        record = await conn.fetch_row(record_id)
    return record
""",
            "control": """def acquire_record(pool, record_id: int) -> dict:
    conn = pool.acquire()
    record = conn.fetch_row(record_id)
    conn.release()
    return record
""",
        },
        {
            "desc": "Async task concurrency aggregation",
            "target": """import asyncio

async def compute_batch(tasks: list) -> list:
    coros = [task.execute() for task in tasks]
    results = await asyncio.gather(*coros)
    return results
""",
            "control": """def compute_batch(tasks: list) -> list:
    results = [task.execute() for task in tasks]
    return results
""",
        },
        {
            "desc": "Async file read buffer",
            "target": """async def read_async_chunk(reader, chunk_size: int) -> bytes:
    chunk = await reader.read(chunk_size)
    header = chunk[:4]
    return header
""",
            "control": """def read_async_chunk(reader, chunk_size: int) -> bytes:
    chunk = reader.read(chunk_size)
    header = chunk[:4]
    return header
""",
        },
        {
            "desc": "Async lock synchronization",
            "target": """import asyncio

async def safe_append(lock: asyncio.Lock, store: list, val: int) -> int:
    async with lock:
        store.append(val)
        length = len(store)
    return length
""",
            "control": """def safe_append(lock, store: list, val: int) -> int:
    lock.acquire()
    store.append(val)
    length = len(store)
    lock.release()
    return length
""",
        },
        {
            "desc": "Async queue get and put workflow",
            "target": """import asyncio

async def process_queue_item(in_q: asyncio.Queue, out_q: asyncio.Queue) -> None:
    item = await in_q.get()
    processed = item.strip().lower()
    await out_q.put(processed)
    in_q.task_done()
""",
            "control": """def process_queue_item(in_q, out_q) -> None:
    item = in_q.get()
    processed = item.strip().lower()
    out_q.put(processed)
    in_q.task_done()
""",
        },
        {
            "desc": "Async event notification wait",
            "target": """import asyncio

async def wait_for_ready(event: asyncio.Event, timeout: float) -> bool:
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False
""",
            "control": """import threading

def wait_for_ready(event: threading.Event, timeout: float) -> bool:
    signaled = event.wait(timeout=timeout)
    return signaled
""",
        },
        {
            "desc": "Async pipeline transformation",
            "target": """async def run_pipeline_step(step_fn, payload: dict) -> dict:
    validated = await step_fn.validate(payload)
    output = await step_fn.transform(validated)
    return output
""",
            "control": """def run_pipeline_step(step_fn, payload: dict) -> dict:
    validated = step_fn.validate(payload)
    output = step_fn.transform(validated)
    return output
""",
        },
    ]

    pairs = []
    for i, t in enumerate(templates, 1):
        pairs.append(
            {
                "pair_id": f"async_await_{i:03d}",
                "category": "async_await",
                "target_construct": "AsyncFunctionDef",
                "diff_summary": t["desc"],
                "target_code": t["target"],
                "control_code": t["control"],
            }
        )
    return pairs


def create_class_def_pairs() -> List[Dict[str, Any]]:
    """Build contrastive pairs for class_def (Target: ast.ClassDef vs Control: dict/closure/module)."""
    templates = [
        {
            "desc": "Stateful counter object vs dictionary closure",
            "target": """class Counter:
    def __init__(self, initial: int = 0):
        self.value = initial

    def increment(self, step: int = 1) -> int:
        self.value += step
        return self.value
""",
            "control": """def make_counter(initial: int = 0):
    state = {"value": initial}
    def increment(step: int = 1) -> int:
        state["value"] += step
        return state["value"]
    return increment
""",
        },
        {
            "desc": "Point2D geometry container vs tuple functions",
            "target": """import math

class Point2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to_origin(self) -> float:
        return math.hypot(self.x, self.y)
""",
            "control": """import math

def make_point(x: float, y: float) -> dict:
    return {"x": x, "y": y}

def point_distance_to_origin(point: dict) -> float:
    return math.hypot(point["x"], point["y"])
""",
        },
        {
            "desc": "Stack data structure vs list wrapper functions",
            "target": """class TokenStack:
    def __init__(self):
        self._items = []

    def push(self, item: str) -> None:
        self._items.append(item)

    def pop(self) -> str:
        return self._items.pop()
""",
            "control": """def create_token_stack() -> list:
    return []

def stack_push(stack: list, item: str) -> None:
    stack.append(item)

def stack_pop(stack: list) -> str:
    return stack.pop()
""",
        },
        {
            "desc": "Hyperparameter configuration container vs dictionary",
            "target": """class ModelConfig:
    def __init__(self, d_model: int, n_layers: int, lr: float):
        self.d_model = d_model
        self.n_layers = n_layers
        self.lr = lr

    def total_parameters(self) -> int:
        return self.d_model * self.n_layers * 12
""",
            "control": """def build_model_config(d_model: int, n_layers: int, lr: float) -> dict:
    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "lr": lr,
        "total_parameters": d_model * n_layers * 12,
    }
""",
        },
        {
            "desc": "Moving average metric tracker vs state dict updater",
            "target": """class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, val: float) -> float:
        self.total += val
        self.count += 1
        return self.total / self.count
""",
            "control": """def create_running_mean() -> dict:
    return {"total": 0.0, "count": 0}

def update_running_mean(state: dict, val: float) -> float:
    state["total"] += val
    state["count"] += 1
    return state["total"] / state["count"]
""",
        },
        {
            "desc": "Tokenizer vocabulary mapper vs lookup function",
            "target": """class VocabMap:
    def __init__(self, tokens: list):
        self.w2i = {tok: i for i, tok in enumerate(tokens)}

    def encode(self, token: str) -> int:
        return self.w2i.get(token, -1)
""",
            "control": """def build_vocab_dict(tokens: list) -> dict:
    return {tok: i for i, tok in enumerate(tokens)}

def encode_token(vocab: dict, token: str) -> int:
    return vocab.get(token, -1)
""",
        },
        {
            "desc": "Activation cache accumulator vs dictionary container",
            "target": """class ActivationCache:
    def __init__(self):
        self.layers = {}

    def store(self, layer_idx: int, act) -> None:
        self.layers[layer_idx] = act
""",
            "control": """def make_activation_store() -> dict:
    return {"layers": {}}

def store_activation(store: dict, layer_idx: int, act) -> None:
    store["layers"][layer_idx] = act
""",
        },
        {
            "desc": "Loss penalty evaluator vs scalar function",
            "target": """class Regularizer:
    def __init__(self, weight: float = 1e-4):
        self.weight = weight

    def penalty(self, norm_val: float) -> float:
        return self.weight * (norm_val ** 2)
""",
            "control": """def compute_penalty(norm_val: float, weight: float = 1e-4) -> float:
    return weight * (norm_val ** 2)
""",
        },
        {
            "desc": "Rate limiter window vs tuple timestamp check",
            "target": """import time

class RateLimiter:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.last_call = 0.0

    def allow(self) -> bool:
        now = time.time()
        if now - self.last_call >= self.interval:
            self.last_call = now
            return True
        return False
""",
            "control": """import time

def allow_call(last_call: float, interval: float = 1.0) -> tuple:
    now = time.time()
    if now - last_call >= interval:
        return True, now
    return False, last_call
""",
        },
        {
            "desc": "Gradient accumulator container vs list reduce",
            "target": """class GradBuffer:
    def __init__(self, steps: int = 4):
        self.steps = steps
        self.grads = []

    def push_grad(self, g: float) -> bool:
        self.grads.append(g)
        return len(self.grads) >= self.steps
""",
            "control": """def push_grad_sync(grads: list, g: float, max_steps: int = 4) -> bool:
    grads.append(g)
    return len(grads) >= max_steps
""",
        },
    ]

    pairs = []
    for i, t in enumerate(templates, 1):
        pairs.append(
            {
                "pair_id": f"class_def_{i:03d}",
                "category": "class_def",
                "target_construct": "ClassDef",
                "diff_summary": t["desc"],
                "target_code": t["target"],
                "control_code": t["control"],
            }
        )
    return pairs


def create_nested_loop_pairs() -> List[Dict[str, Any]]:
    """Build contrastive pairs for nested_loop (Target: 2D nested loops vs Control: flat 1D loop)."""
    templates = [
        {
            "desc": "Matrix coordinate grid sum",
            "target": """def sum_matrix_grid(rows: int, cols: int) -> int:
    total = 0
    for r in range(rows):
        for c in range(cols):
            total += (r * cols + c)
    return total
""",
            "control": """def sum_matrix_grid(rows: int, cols: int) -> int:
    total = 0
    for idx in range(rows * cols):
        total += idx
    return total
""",
        },
        {
            "desc": "Pairwise token interaction cross product",
            "target": """def compute_pairwise_matches(tokens_a: list, tokens_b: list) -> int:
    matches = 0
    for a in tokens_a:
        for b in tokens_b:
            if a == b:
                matches += 1
    return matches
""",
            "control": """def compute_pairwise_matches(tokens_a: list, tokens_b: list) -> int:
    set_b = set(tokens_b)
    matches = 0
    for a in tokens_a:
        if a in set_b:
            matches += 1
    return matches
""",
        },
        {
            "desc": "Nested list flattening",
            "target": """def flatten_nested_lists(matrix: list) -> list:
    flat = []
    for row in matrix:
        for val in row:
            flat.append(val)
    return flat
""",
            "control": """def flatten_nested_lists(matrix: list) -> list:
    flat = []
    for row in matrix:
        flat.extend(row)
    return flat
""",
        },
        {
            "desc": "Attention weight matrix mask zeroing",
            "target": """def zero_lower_triangle(n: int) -> list:
    grid = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            grid[i][j] = -1e9
    return grid
""",
            "control": """def zero_lower_triangle(n: int) -> list:
    grid = [[0.0] * n for _ in range(n)]
    for idx in range(n * n):
        i = idx // n
        j = idx % n
        if j < i:
            grid[i][j] = -1e9
    return grid
""",
        },
        {
            "desc": "2D convolution sliding window dot product",
            "target": """def convolve_2d_window(image: list, kernel: list) -> float:
    acc = 0.0
    for i in range(3):
        for j in range(3):
            acc += image[i][j] * kernel[i][j]
    return acc
""",
            "control": """def convolve_2d_window(image: list, kernel: list) -> float:
    acc = 0.0
    for idx in range(9):
        i = idx // 3
        j = idx % 3
        acc += image[i][j] * kernel[i][j]
    return acc
""",
        },
        {
            "desc": "Find coordinate of target element in 2D grid",
            "target": """def find_cell(grid: list, target_val: int) -> tuple:
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == target_val:
                return (r, c)
    return (-1, -1)
""",
            "control": """def find_cell(grid: list, target_val: int) -> tuple:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        if grid[r][c] == target_val:
            return (r, c)
    return (-1, -1)
""",
        },
        {
            "desc": "Nested while loop counter",
            "target": """def nested_while_steps(limit_i: int, limit_j: int) -> int:
    steps = 0
    i = 0
    while i < limit_i:
        j = 0
        while j < limit_j:
            steps += 1
            j += 1
        i += 1
    return steps
""",
            "control": """def nested_while_steps(limit_i: int, limit_j: int) -> int:
    steps = 0
    idx = 0
    total = limit_i * limit_j
    while idx < total:
        steps += 1
        idx += 1
    return steps
""",
        },
        {
            "desc": "Multi-head attention projection indexing",
            "target": """def project_heads(n_heads: int, head_dim: int) -> list:
    indices = []
    for h in range(n_heads):
        for d in range(head_dim):
            indices.append(h * head_dim + d)
    return indices
""",
            "control": """def project_heads(n_heads: int, head_dim: int) -> list:
    total_dim = n_heads * head_dim
    indices = [i for i in range(total_dim)]
    return indices
""",
        },
        {
            "desc": "Nested for-loop Cartesian product tuple generation",
            "target": """def cartesian_coords(xs: list, ys: list) -> list:
    points = []
    for x in xs:
        for y in ys:
            points.append((x, y))
    return points
""",
            "control": """import itertools

def cartesian_coords(xs: list, ys: list) -> list:
    points = list(itertools.product(xs, ys))
    return points
""",
        },
        {
            "desc": "Matrix diagonal boundary iteration",
            "target": """def count_off_diagonals(size: int) -> int:
    count = 0
    for r in range(size):
        for c in range(size):
            if r != c:
                count += 1
    return count
""",
            "control": """def count_off_diagonals(size: int) -> int:
    count = 0
    for idx in range(size * size):
        r, c = divmod(idx, size)
        if r != c:
            count += 1
    return count
""",
        },
    ]

    pairs = []
    for i, t in enumerate(templates, 1):
        pairs.append(
            {
                "pair_id": f"nested_loop_{i:03d}",
                "category": "nested_loop",
                "target_construct": "NestedLoop",
                "diff_summary": t["desc"],
                "target_code": t["target"],
                "control_code": t["control"],
            }
        )
    return pairs


def build_and_validate_corpus(output_path: Path) -> List[Dict[str, Any]]:
    """Assemble all 5 categories, validate AST parsing, and write JSONL."""
    all_pairs: List[Dict[str, Any]] = []
    all_pairs.extend(create_try_except_pairs())
    all_pairs.extend(create_with_statement_pairs())
    all_pairs.extend(create_async_await_pairs())
    all_pairs.extend(create_class_def_pairs())
    all_pairs.extend(create_nested_loop_pairs())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    verified_records = []
    for item in all_pairs:
        # 1. Native AST validation
        validate_python_code(item["target_code"], f"{item['pair_id']} (target)")
        validate_python_code(item["control_code"], f"{item['pair_id']} (control)")

        # 2. Extract AST Node class names
        target_nodes = extract_ast_nodes(item["target_code"])
        control_nodes = extract_ast_nodes(item["control_code"])

        record = {
            "pair_id": item["pair_id"],
            "category": item["category"],
            "target_construct": item["target_construct"],
            "diff_summary": item["diff_summary"],
            "target_code": item["target_code"],
            "control_code": item["control_code"],
            "target_ast_nodes": target_nodes,
            "control_ast_nodes": control_nodes,
            "target_char_len": len(item["target_code"]),
            "control_char_len": len(item["control_code"]),
            "target_valid": True,
            "control_valid": True,
        }
        verified_records.append(record)

    # Write JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in verified_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return verified_records


def main():
    root_dir = Path(__file__).resolve().parent.parent
    corpus_file = root_dir / "corpus" / "gemma_ast_corpus.jsonl"

    print(f"Generating synthetic AST corpus at: {corpus_file}")
    records = build_and_validate_corpus(corpus_file)

    categories = set(r["category"] for r in records)
    print(f"\nSuccessfully generated & validated {len(records)} contrastive snippet pairs:")
    for cat in sorted(categories):
        count = sum(1 for r in records if r["category"] == cat)
        print(f"  - {cat:16s}: {count} pairs")

    print(f"\nCorpus saved to: {corpus_file}")


if __name__ == "__main__":
    main()
