"""Synthetic AST Corpus Generator for Mechanistic Interpretability.

Generates 60 contrastive Python snippet pairs per category across 5 AST constructs:
1. try_except     (60 pairs: ast.Try vs. conditional guards / status checks)
2. with_statement (60 pairs: ast.With vs. manual acquire-try-release / explicit close)
3. async_await    (60 pairs: ast.AsyncFunctionDef/Await vs. synchronous equivalents)
4. class_def      (60 pairs: ast.ClassDef vs. closures / dictionary state / functional modules)
5. nested_loop    (60 pairs: nested ast.For/While vs. flat 1D indexing / single loop passes)

Total: 300 contrastive pairs = 600 validated snippets.
Outputs are saved to `corpus/gemma_ast_corpus.jsonl`.
"""

import ast
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Set

SEED = 42
random.seed(SEED)


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


# ==============================================================================
# CATEGORY 1: TRY_EXCEPT (60 Pairs)
# ==============================================================================
def create_try_except_pairs() -> List[Dict[str, Any]]:
    pairs = []
    
    # Family 1: Math & Arithmetic Guards (12 pairs)
    math_specs = [
        ("divide_floats", "float", "numerator", "denominator", "numerator / denominator", "denominator == 0.0", "0.0", "ZeroDivisionError"),
        ("modulo_integers", "int", "dividend", "divisor", "dividend % divisor", "divisor == 0", "0", "ZeroDivisionError"),
        ("floor_divide", "int", "total_items", "batch_size", "total_items // batch_size", "batch_size <= 0", "0", "ZeroDivisionError"),
        ("square_root_eval", "float", "radicand", "base_val", "math.sqrt(radicand) + base_val", "radicand < 0.0", "base_val", "ValueError"),
        ("logarithm_calc", "float", "input_signal", "offset", "math.log(input_signal) + offset", "input_signal <= 0.0", "offset", "ValueError"),
        ("power_exponent", "float", "base_num", "exponent", "math.pow(base_num, exponent)", "base_num < 0 and not exponent.is_integer()", "0.0", "ValueError"),
        ("factorial_eval", "int", "n_elements", "dummy", "math.factorial(n_elements)", "n_elements < 0", "0", "ValueError"),
        ("asin_angle", "float", "ratio", "default_rad", "math.asin(ratio)", "ratio < -1.0 or ratio > 1.0", "default_rad", "ValueError"),
        ("acos_angle", "float", "ratio", "default_rad", "math.acos(ratio)", "ratio < -1.0 or ratio > 1.0", "default_rad", "ValueError"),
        ("gamma_calc", "float", "x_val", "fallback", "math.gamma(x_val)", "x_val <= 0 and x_val.is_integer()", "fallback", "ValueError"),
        ("fmod_eval", "float", "coord_x", "coord_y", "math.fmod(coord_x, coord_y)", "coord_y == 0.0", "0.0", "ZeroDivisionError"),
        ("reciprocal_rate", "float", "duration_sec", "scale", "scale / duration_sec", "duration_sec <= 0.0", "0.0", "ZeroDivisionError"),
    ]
    for fn, typ, a1, a2, op, guard, fallback, exc in math_specs:
        target = f"""import math

def {fn}({a1}: {typ}, {a2}: {typ}) -> {typ}:
    try:
        result = {op}
        return result
    except {exc}:
        return {fallback}
"""
        control = f"""import math

def {fn}({a1}: {typ}, {a2}: {typ}) -> {typ}:
    if {guard}:
        return {fallback}
    else:
        result = {op}
        return result
"""
        pairs.append({
            "category": "try_except",
            "target_construct": "Try",
            "diff_summary": f"Math error handling vs guard: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 2: Type Casts & String Conversions (12 pairs)
    conv_specs = [
        ("parse_int_token", "str", "raw_token", "int", "int(raw_token)", "raw_token.lstrip('-').isdigit()", "0", "ValueError"),
        ("parse_float_weight", "str", "raw_weight", "float", "float(raw_weight)", "raw_weight.replace('.', '', 1).lstrip('-').isdigit()", "0.0", "ValueError"),
        ("parse_bool_flag", "str", "raw_flag", "bool", "bool(int(raw_flag))", "raw_flag in ('0', '1')", "False", "ValueError"),
        ("parse_hex_code", "str", "hex_str", "int", "int(hex_str, 16)", "all(c in '0123456789abcdefABCDEF' for c in hex_str) and len(hex_str) > 0", "-1", "ValueError"),
        ("parse_binary_mask", "str", "bin_str", "int", "int(bin_str, 2)", "all(c in '01' for c in bin_str) and len(bin_str) > 0", "0", "ValueError"),
        ("parse_octal_perm", "str", "oct_str", "int", "int(oct_str, 8)", "all(c in '01234567' for c in oct_str) and len(oct_str) > 0", "0", "ValueError"),
        ("str_to_complex", "str", "cmplx_str", "complex", "complex(cmplx_str)", "('+' in cmplx_str or '-' in cmplx_str) and 'j' in cmplx_str", "0j", "ValueError"),
        ("convert_loss_str", "str", "loss_repr", "float", "float(loss_repr)", "len(loss_repr) > 0 and loss_repr.replace('.', '', 1).isdigit()", "999.0", "ValueError"),
        ("timestamp_to_int", "str", "ts_token", "int", "int(ts_token)", "ts_token.isdigit()", "0", "ValueError"),
        ("epoch_parse", "str", "epoch_str", "int", "int(epoch_str)", "epoch_str.isdigit()", "1", "ValueError"),
        ("layer_idx_parse", "str", "idx_str", "int", "int(idx_str)", "idx_str.isdigit()", "-1", "ValueError"),
        ("token_id_parse", "str", "tid_str", "int", "int(tid_str)", "tid_str.isdigit()", "0", "ValueError"),
    ]
    for fn, in_t, arg, out_t, op, guard, fallback, exc in conv_specs:
        target = f"""def {fn}({arg}: {in_t}) -> {out_t}:
    try:
        val = {op}
        return val
    except {exc}:
        return {fallback}
"""
        control = f"""def {fn}({arg}: {in_t}) -> {out_t}:
    if {guard}:
        val = {op}
        return val
    else:
        return {fallback}
"""
        pairs.append({
            "category": "try_except",
            "target_construct": "Try",
            "diff_summary": f"Type cast fallback vs conditional check: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 3: Container Indexing & Map Lookups (12 pairs)
    container_specs = [
        ("get_dict_setting", "dict", "config", "str", "key", "config[key]", "key in config", "None", "KeyError"),
        ("get_nested_val", "dict", "payload", "str", "field", "payload['data'][field]", "'data' in payload and field in payload['data']", "''", "KeyError"),
        ("fetch_cache_item", "dict", "cache", "str", "token_key", "cache[token_key]", "token_key in cache", "{}", "KeyError"),
        ("lookup_vocab_id", "dict", "vocab", "str", "word", "vocab[word]", "word in vocab", "-1", "KeyError"),
        ("lookup_metadata", "dict", "meta", "str", "tag", "meta[tag]", "tag in meta", "'default'", "KeyError"),
        ("lookup_hyperparam", "dict", "hparams", "str", "name", "hparams[name]", "name in hparams", "0.0", "KeyError"),
        ("get_first_item", "list", "records", "int", "idx", "records[idx]", "0 <= idx < len(records)", "None", "IndexError"),
        ("get_last_token", "list", "tokens", "int", "offset", "tokens[-offset]", "0 < offset <= len(tokens)", "'<PAD>'", "IndexError"),
        ("get_matrix_row", "list", "matrix", "int", "r_idx", "matrix[r_idx]", "0 <= r_idx < len(matrix)", "[]", "IndexError"),
        ("pop_fifo_queue", "list", "queue", "int", "pos", "queue.pop(pos)", "0 <= pos < len(queue)", "'EMPTY'", "IndexError"),
        ("lookup_tensor_shape", "list", "dims", "int", "dim_i", "dims[dim_i]", "0 <= dim_i < len(dims)", "1", "IndexError"),
        ("fetch_sample_batch", "list", "batches", "int", "b_idx", "batches[b_idx]", "0 <= b_idx < len(batches)", "{}", "IndexError"),
    ]
    for fn, c_t, c_name, a_t, a_name, op, guard, fallback, exc in container_specs:
        target = f"""def {fn}({c_name}: {c_t}, {a_name}: {a_t}):
    try:
        val = {op}
        return val
    except {exc}:
        return {fallback}
"""
        control = f"""def {fn}({c_name}: {c_t}, {a_name}: {a_t}):
    if {guard}:
        val = {op}
        return val
    else:
        return {fallback}
"""
        pairs.append({
            "category": "try_except",
            "target_construct": "Try",
            "diff_summary": f"Container access vs membership/bounds check: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 4: Attribute, Iterator & Object Introspection (12 pairs)
    obj_specs = [
        ("get_module_attr", "module_obj", "__version__", "module_obj.__version__", "hasattr(module_obj, '__version__')", "'unknown'", "AttributeError"),
        ("get_model_config", "model_obj", "hidden_dim", "model_obj.hidden_dim", "hasattr(model_obj, 'hidden_dim')", "768", "AttributeError"),
        ("get_layer_weights", "layer_obj", "weight", "layer_obj.weight", "hasattr(layer_obj, 'weight')", "None", "AttributeError"),
        ("get_tokenizer_pad", "tok_obj", "pad_token_id", "tok_obj.pad_token_id", "hasattr(tok_obj, 'pad_token_id')", "0", "AttributeError"),
        ("get_optimizer_lr", "opt_obj", "lr", "opt_obj.lr", "hasattr(opt_obj, 'lr')", "0.001", "AttributeError"),
        ("get_dataset_split", "ds_obj", "split", "ds_obj.split", "hasattr(ds_obj, 'split')", "'train'", "AttributeError"),
        ("get_hook_name", "hook_obj", "name", "hook_obj.name", "hasattr(hook_obj, 'name')", "'unnamed_hook'", "AttributeError"),
        ("get_device_type", "tensor_obj", "device", "str(tensor_obj.device)", "hasattr(tensor_obj, 'device')", "'cpu'", "AttributeError"),
        ("advance_iterator", "it_obj", "next_item", "next(it_obj)", "hasattr(it_obj, '__next__')", "None", "StopIteration"),
        ("read_first_chunk", "stream_obj", "chunk", "next(stream_obj)", "hasattr(stream_obj, '__iter__')", "'EOF'", "StopIteration"),
        ("get_head_dim", "layer_obj", "head_dim", "layer_obj.head_dim", "hasattr(layer_obj, 'head_dim')", "64", "AttributeError"),
        ("get_vocab_size", "model_obj", "vocab_size", "model_obj.vocab_size", "hasattr(model_obj, 'vocab_size')", "32000", "AttributeError"),
    ]
    for fn, obj_n, attr, op, guard, fallback, exc in obj_specs:
        target = f"""def {fn}({obj_n}: object):
    try:
        val = {op}
        return val
    except {exc}:
        return {fallback}
"""
        control = f"""def {fn}({obj_n}: object):
    if {guard}:
        val = {op}
        return val
    else:
        return {fallback}
"""
        pairs.append({
            "category": "try_except",
            "target_construct": "Try",
            "diff_summary": f"Attribute/iter access vs introspection guard: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 5: File & Serialization Parsing (12 pairs)
    io_specs = [
        ("read_text_lines", "filepath: str", "open(filepath, 'r').readlines()", "os.path.isfile(filepath)", "[]", "FileNotFoundError"),
        ("load_json_data", "raw_json: str", "json.loads(raw_json)", "raw_json.startswith('{') and raw_json.endswith('}')", "{}", "json.JSONDecodeError"),
        ("remove_temp_file", "filepath: str", "os.remove(filepath)", "os.path.exists(filepath)", "None", "FileNotFoundError"),
        ("read_binary_blob", "filepath: str", "open(filepath, 'rb').read()", "os.path.exists(filepath)", "b''", "FileNotFoundError"),
        ("parse_csv_header", "filepath: str", "open(filepath, 'r').readline().strip()", "os.path.exists(filepath)", "''", "FileNotFoundError"),
        ("decode_utf8_bytes", "blob: bytes", "blob.decode('utf-8')", "isinstance(blob, bytes) and len(blob) > 0", "''", "UnicodeDecodeError"),
        ("decode_ascii_str", "blob: bytes", "blob.decode('ascii')", "all(b < 128 for b in blob)", "''", "UnicodeDecodeError"),
        ("read_yaml_config", "content: str", "json.loads(content)", "len(content.strip()) > 0", "{}", "Exception"),
        ("open_lockfile", "lockfile: str", "open(lockfile, 'x')", "not os.path.exists(lockfile)", "None", "FileExistsError"),
        ("make_sub_directory", "dirpath: str", "os.mkdir(dirpath)", "not os.path.exists(dirpath)", "None", "FileExistsError"),
        ("stat_file_size", "filepath: str", "os.stat(filepath).st_size", "os.path.exists(filepath)", "0", "FileNotFoundError"),
        ("list_dir_entries", "dirpath: str", "os.listdir(dirpath)", "os.path.isdir(dirpath)", "[]", "FileNotFoundError"),
    ]
    for fn, arg, op, guard, fallback, exc in io_specs:
        target = f"""import os
import json

def {fn}({arg}):
    try:
        res = {op}
        return res
    except {exc}:
        return {fallback}
"""
        control = f"""import os
import json

def {fn}({arg}):
    if {guard}:
        res = {op}
        return res
    else:
        return {fallback}
"""
        pairs.append({
            "category": "try_except",
            "target_construct": "Try",
            "diff_summary": f"IO operation vs existence check: {fn}",
            "target_code": target,
            "control_code": control,
        })

    return pairs[:60]


# ==============================================================================
# CATEGORY 2: WITH_STATEMENT (60 Pairs)
# ==============================================================================
def create_with_statement_pairs() -> List[Dict[str, Any]]:
    pairs = []
    
    # Family 1: File I/O text & binary contexts (15 pairs)
    modes = [
        ("read_text_file", "r", "handle.read()", "str", "", "data = handle.read()"),
        ("read_line_list", "r", "handle.readlines()", "list", "", "data = handle.readlines()"),
        ("write_text_summary", "w", "handle.write(payload)", "int", "payload: str", "data = handle.write(payload)"),
        ("append_log_message", "a", "handle.write(msg + '\\n')", "int", "msg: str", "data = handle.write(msg + '\\n')"),
        ("read_raw_bytes", "rb", "handle.read()", "bytes", "", "data = handle.read()"),
        ("write_binary_blob", "wb", "handle.write(blob)", "int", "blob: bytes", "data = handle.write(blob)"),
        ("append_binary_record", "ab", "handle.write(record)", "int", "record: bytes", "data = handle.write(record)"),
        ("read_first_line", "r", "handle.readline()", "str", "", "data = handle.readline()"),
        ("read_chunked_text", "r", "handle.read(1024)", "str", "", "data = handle.read(1024)"),
        ("write_json_dump", "w", "handle.write(json_str)", "int", "json_str: str", "data = handle.write(json_str)"),
        ("read_csv_table", "r", "handle.read().splitlines()", "list", "", "data = handle.read().splitlines()"),
        ("write_audit_trail", "a", "handle.write(event_row)", "int", "event_row: str", "data = handle.write(event_row)"),
        ("read_byte_header", "rb", "handle.read(16)", "bytes", "", "data = handle.read(16)"),
        ("dump_model_weights", "wb", "handle.write(weight_bytes)", "int", "weight_bytes: bytes", "data = handle.write(weight_bytes)"),
        ("write_config_ini", "w", "handle.write(cfg_text)", "int", "cfg_text: str", "data = handle.write(cfg_text)"),
    ]
    for fn, mode, action, ret_t, extra_arg, exec_stmt in modes:
        arg_sig = f"filepath: str, {extra_arg}" if extra_arg else "filepath: str"
        target = f"""def {fn}({arg_sig}) -> {ret_t}:
    with open(filepath, "{mode}") as handle:
        {exec_stmt}
    return data
"""
        control = f"""def {fn}({arg_sig}) -> {ret_t}:
    handle = open(filepath, "{mode}")
    {exec_stmt}
    handle.close()
    return data
"""
        pairs.append({
            "category": "with_statement",
            "target_construct": "With",
            "diff_summary": f"File I/O context vs explicit close: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 2: Threading & Concurrency Locks (15 pairs)
    lock_specs = [
        ("sync_increment_counter", "state['count'] += delta; res = state['count']", "delta: int", "int"),
        ("sync_append_history", "history.append(entry); res = len(history)", "entry: str", "int"),
        ("sync_pop_work_item", "res = queue.pop(0) if queue else None", "", "object"),
        ("sync_update_weight", "model_params['lr'] = new_lr; res = model_params['lr']", "new_lr: float", "float"),
        ("sync_set_flag", "flags['ready'] = True; res = True", "", "bool"),
        ("sync_clear_cache", "cache.clear(); res = 0", "", "int"),
        ("sync_store_metric", "metrics[name] = val; res = val", "name: str, val: float", "float"),
        ("sync_record_latency", "latencies.append(ms); res = sum(latencies)", "ms: float", "float"),
        ("sync_swap_buffer", "active_buf, idle_buf = idle_buf, active_buf; res = len(active_buf)", "", "int"),
        ("sync_update_step", "state['step'] += 1; res = state['step']", "", "int"),
        ("sync_decrement_ref", "state['refs'] -= 1; res = state['refs']", "", "int"),
        ("sync_push_token", "tokens.append(t); res = len(tokens)", "t: int", "int"),
        ("sync_set_gradient", "grads[param_id] = grad_val; res = True", "param_id: str, grad_val: float", "bool"),
        ("sync_update_timestamp", "meta['updated'] = ts; res = ts", "ts: float", "float"),
        ("sync_reset_tracker", "tracker['running'] = False; res = False", "", "bool"),
    ]
    for i, (fn, body, extra_p, ret_t) in enumerate(lock_specs):
        sig = f"lock, state: dict, {extra_p}" if extra_p else "lock, state: dict"
        target = f"""def {fn}({sig}) -> {ret_t}:
    with lock:
        {body}
    return res
"""
        control = f"""def {fn}({sig}) -> {ret_t}:
    lock.acquire()
    {body}
    lock.release()
    return res
"""
        pairs.append({
            "category": "with_statement",
            "target_construct": "With",
            "diff_summary": f"Lock context manager vs acquire/release: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 3: Contextlib & State Management Contexts (15 pairs)
    ctx_specs = [
        ("redirect_stdout_buf", "io.StringIO()", "redirect_stdout(buf)", "buf = io.StringIO(); old = sys.stdout; sys.stdout = buf", "sys.stdout = old; res = buf.getvalue()", "buf.getvalue()"),
        ("redirect_stderr_buf", "io.StringIO()", "redirect_stderr(buf)", "buf = io.StringIO(); old = sys.stderr; sys.stderr = buf", "sys.stderr = old; res = buf.getvalue()", "buf.getvalue()"),
        ("suppress_key_error", "None", "suppress(KeyError)", "pass", "res = mapping.get('key', None)", "mapping.get('key', None)"),
        ("suppress_val_error", "None", "suppress(ValueError)", "pass", "res = int(val_str) if val_str.isdigit() else 0", "int(val_str) if val_str.isdigit() else 0"),
        ("temp_cwd_context", "os.getcwd()", "temp_chdir(target_dir)", "old_dir = os.getcwd(); os.chdir(target_dir)", "os.chdir(old_dir); res = target_dir", "target_dir"),
        ("torch_no_grad_scope", "None", "torch.no_grad()", "prev = torch.is_grad_enabled(); torch.set_grad_enabled(False)", "torch.set_grad_enabled(prev); res = model(x)", "model(x)"),
        ("torch_inference_scope", "None", "torch.inference_mode()", "prev = torch.is_inference_mode_enabled(); torch.set_inference_mode(True)", "torch.set_inference_mode(prev); res = model(x)", "model(x)"),
        ("decimal_context_scope", "None", "decimal.localcontext()", "old_prec = decimal.getcontext().prec; decimal.getcontext().prec = 28", "decimal.getcontext().prec = old_prec; res = decimal.Decimal(1) / 7", "decimal.Decimal(1) / 7"),
        ("sqlite_transaction", "None", "connection", "connection.execute('BEGIN')", "connection.execute('COMMIT'); res = True", "True"),
        ("zipfile_reader", "None", "zipfile.ZipFile(archive_path, 'r')", "zf = zipfile.ZipFile(archive_path, 'r')", "zf.close(); res = file_list", "file_list"),
        ("tarfile_reader", "None", "tarfile.open(archive_path, 'r')", "tf = tarfile.open(archive_path, 'r')", "tf.close(); res = member_list", "member_list"),
        ("temp_directory_scope", "None", "tempfile.TemporaryDirectory()", "temp_dir = tempfile.mkdtemp()", "shutil.rmtree(temp_dir); res = temp_dir", "temp_dir"),
        ("socket_connection_scope", "None", "socket.create_connection((host, port))", "sock = socket.create_connection((host, port))", "sock.close(); res = True", "True"),
        ("database_cursor_scope", "None", "connection.cursor()", "cur = connection.cursor()", "cur.close(); res = rows", "rows"),
        ("temp_named_file", "None", "tempfile.NamedTemporaryFile()", "tf = tempfile.NamedTemporaryFile(); tf_name = tf.name", "tf.close(); res = tf_name", "tf_name"),
    ]
    for fn, init, with_expr, manual_enter, manual_exit, ret_expr in ctx_specs:
        target = f"""def {fn}(resource, x=None):
    with {with_expr} as handle:
        res = {ret_expr}
    return res
"""
        control = f"""def {fn}(resource, x=None):
    {manual_enter}
    {manual_exit}
    return res
"""
        pairs.append({
            "category": "with_statement",
            "target_construct": "With",
            "diff_summary": f"Context manager scope vs manual setup/teardown: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 4: Custom Resource & Buffer Handlers (15 pairs)
    for idx in range(1, 16):
        fn = f"manage_pipeline_resource_{idx:02d}"
        target = f"""def {fn}(manager, config: dict) -> dict:
    with manager.session(config) as active_sess:
        output = active_sess.execute_step({idx})
    return output
"""
        control = f"""def {fn}(manager, config: dict) -> dict:
    active_sess = manager.session(config)
    active_sess.open()
    output = active_sess.execute_step({idx})
    active_sess.close()
    return output
"""
        pairs.append({
            "category": "with_statement",
            "target_construct": "With",
            "diff_summary": f"Pipeline resource context manager vs open/close lifecycle: {fn}",
            "target_code": target,
            "control_code": control,
        })

    return pairs[:60]


# ==============================================================================
# CATEGORY 3: ASYNC_AWAIT (60 Pairs)
# ==============================================================================
def create_async_await_pairs() -> List[Dict[str, Any]]:
    pairs = []
    
    # Family 1: Asynchronous HTTP & Network Requests (15 pairs)
    http_methods = [
        ("fetch_json_endpoint", "client.get(url)", "client.get(url)", "response.json()", "response.json()", "url: str"),
        ("post_data_payload", "client.post(url, json=payload)", "client.post(url, json=payload)", "response.text()", "response.text()", "url: str, payload: dict"),
        ("put_update_resource", "client.put(url, json=data)", "client.put(url, json=data)", "response.status_code", "response.status_code", "url: str, data: dict"),
        ("delete_remote_item", "client.delete(url)", "client.delete(url)", "response.ok", "response.ok", "url: str"),
        ("head_check_header", "client.head(url)", "client.head(url)", "response.headers.get('ETag')", "response.headers.get('ETag')", "url: str"),
        ("patch_field_delta", "client.patch(url, json=delta)", "client.patch(url, json=delta)", "response.json()", "response.json()", "url: str, delta: dict"),
        ("fetch_raw_stream", "client.get(url, stream=True)", "client.get(url, stream=True)", "response.read()", "response.read()", "url: str"),
        ("post_auth_credentials", "client.post(url, data=auth)", "client.post(url, data=auth)", "response.cookies", "response.cookies", "url: str, auth: dict"),
        ("fetch_paged_results", "client.get(f'{url}?page={page}')", "client.get(f'{url}?page={page}')", "response.json()['items']", "response.json()['items']", "url: str, page: int"),
        ("fetch_health_status", "client.get('/health')", "client.get('/health')", "response.status == 200", "response.status == 200", ""),
        ("query_graphql_api", "client.post(url, json={'query': q})", "client.post(url, json={'query': q})", "response.json()['data']", "response.json()['data']", "url: str, q: str"),
        ("fetch_model_manifest", "client.get(f'{repo_id}/manifest')", "client.get(f'{repo_id}/manifest')", "response.text()", "response.text()", "repo_id: str"),
        ("download_checkpoint_chunk", "client.get(url, headers={'Range': rng})", "client.get(url, headers={'Range': rng})", "response.content", "response.content", "url: str, rng: str"),
        ("ping_gateway_service", "client.get('/ping')", "client.get('/ping')", "response.elapsed_ms", "response.elapsed_ms", ""),
        ("upload_multipart_file", "client.post(url, files=files)", "client.post(url, files=files)", "response.status_code == 201", "response.status_code == 201", "url: str, files: dict"),
    ]
    for fn, a_req, s_req, a_res, s_res, extra_args in http_methods:
        sig = f"client, {extra_args}" if extra_args else "client"
        target = f"""async def {fn}({sig}):
    response = await {a_req}
    data = await {a_res}
    return data
"""
        control = f"""def {fn}({sig}):
    response = {s_req}
    data = {s_res}
    return data
"""
        pairs.append({
            "category": "async_await",
            "target_construct": "AsyncFunctionDef",
            "diff_summary": f"Async HTTP request vs synchronous call: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 2: Async Delays, Polling & Event Loops (15 pairs)
    polling_specs = [
        ("poll_job_completion", "asyncio.sleep(interval)", "time.sleep(interval)", "await client.get_job_status(job_id)", "client.get_job_status(job_id)", "job_id: str, interval: float = 0.5"),
        ("poll_training_step", "asyncio.sleep(0.1)", "time.sleep(0.1)", "await monitor.current_loss()", "monitor.current_loss()", "monitor"),
        ("poll_cache_warmup", "asyncio.sleep(0.05)", "time.sleep(0.05)", "await cache.is_ready()", "cache.is_ready()", "cache"),
        ("poll_lock_availability", "asyncio.sleep(0.01)", "time.sleep(0.01)", "await locker.check()", "locker.check()", "locker"),
        ("poll_device_queue", "asyncio.sleep(0.2)", "time.sleep(0.2)", "await device.pending_ops()", "device.pending_ops()", "device"),
        ("poll_worker_heartbeat", "asyncio.sleep(1.0)", "time.sleep(1.0)", "await registry.heartbeat(worker_id)", "registry.heartbeat(worker_id)", "registry, worker_id: str"),
        ("poll_dataset_shard", "asyncio.sleep(0.1)", "time.sleep(0.1)", "await loader.has_next()", "loader.has_next()", "loader"),
        ("poll_eval_checkpoint", "asyncio.sleep(2.0)", "time.sleep(2.0)", "await evaluator.poll_best()", "evaluator.poll_best()", "evaluator"),
        ("poll_stream_eos", "asyncio.sleep(0.02)", "time.sleep(0.02)", "await stream.is_eos()", "stream.is_eos()", "stream"),
        ("poll_buffer_drain", "asyncio.sleep(0.05)", "time.sleep(0.05)", "await buf.is_empty()", "buf.is_empty()", "buf"),
        ("wait_for_condition", "asyncio.sleep(period)", "time.sleep(period)", "await check_fn()", "check_fn()", "check_fn, period: float = 0.1"),
        ("poll_replication_lag", "asyncio.sleep(0.5)", "time.sleep(0.5)", "await replica.lag_seconds()", "replica.lag_seconds()", "replica"),
        ("poll_token_bucket", "asyncio.sleep(0.05)", "time.sleep(0.05)", "await bucket.consume(1)", "bucket.consume(1)", "bucket"),
        ("poll_batch_assembly", "asyncio.sleep(0.01)", "time.sleep(0.01)", "await queue.size() >= batch_size", "queue.size() >= batch_size", "queue, batch_size: int"),
        ("poll_pipeline_flush", "asyncio.sleep(0.1)", "time.sleep(0.1)", "await pipe.flush_done()", "pipe.flush_done()", "pipe"),
    ]
    for fn, a_sleep, s_sleep, a_step, s_step, extra_args in polling_specs:
        target = f"""import asyncio

async def {fn}({extra_args}):
    for _ in range(10):
        await {a_sleep}
        status = {a_step}
        if status:
            return True
    return False
"""
        control = f"""import time

def {fn}({extra_args}):
    for _ in range(10):
        {s_sleep}
        status = {s_step}
        if status:
            return True
    return False
"""
        pairs.append({
            "category": "async_await",
            "target_construct": "AsyncFunctionDef",
            "diff_summary": f"Async polling with asyncio.sleep vs time.sleep: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 3: Async Queue / Stream Producers & Consumers (15 pairs)
    queue_specs = [
        ("drain_queue_items", "q.get()", "q.get()", "q.task_done()", "q.task_done()"),
        ("feed_queue_item", "q.put(item)", "q.put(item)", "pass", "pass"),
        ("transfer_queue_batch", "in_q.get()", "in_q.get()", "out_q.put(val)", "out_q.put(val)"),
        ("filter_queue_stream", "in_q.get()", "in_q.get()", "out_q.put(val)", "out_q.put(val)"),
        ("consume_token_stream", "stream.next_token()", "stream.next_token()", "collector.append(t)", "collector.append(t)"),
        ("read_packet_frame", "reader.read_frame()", "reader.read_frame()", "pass", "pass"),
        ("yield_batch_sample", "sampler.sample()", "sampler.sample()", "pass", "pass"),
        ("forward_event_bus", "bus.receive()", "bus.receive()", "bus.dispatch(evt)", "bus.dispatch(evt)"),
        ("collect_stream_text", "stream.read_chunk()", "stream.read_chunk()", "chunks.append(c)", "chunks.append(c)"),
        ("pipeline_enqueue", "pipe.push(elem)", "pipe.push(elem)", "pass", "pass"),
        ("dequeue_inference_req", "queue.get_request()", "queue.get_request()", "pass", "pass"),
        ("stream_token_chunk", "generator.next_chunk()", "generator.next_chunk()", "pass", "pass"),
        ("flush_stream_buffer", "stream.flush()", "stream.flush()", "pass", "pass"),
        ("receive_ws_message", "socket.recv()", "socket.recv()", "pass", "pass"),
        ("send_ws_acknowledgement", "socket.send('ACK')", "socket.send('ACK')", "pass", "pass"),
    ]
    for idx, (fn, a_act, s_act, a_done, s_done) in enumerate(queue_specs, 1):
        target = f"""async def {fn}_{idx:02d}(q, count: int = 5) -> list:
    items = []
    for _ in range(count):
        val = await {a_act}
        items.append(val)
        {a_done}
    return items
"""
        control = f"""def {fn}_{idx:02d}(q, count: int = 5) -> list:
    items = []
    for _ in range(count):
        val = {s_act}
        items.append(val)
        {s_done}
    return items
"""
        pairs.append({
            "category": "async_await",
            "target_construct": "AsyncFunctionDef",
            "diff_summary": f"Async stream consumption vs synchronous loop: {fn}_{idx:02d}",
            "target_code": target,
            "control_code": control,
        })

    # Family 4: Async Concurrency & Batch Aggregation (15 pairs)
    for idx in range(1, 16):
        fn = f"async_batch_pipeline_{idx:02d}"
        target = f"""import asyncio

async def {fn}(workers: list, payload: dict) -> list:
    tasks = [worker.process(payload) for worker in workers]
    results = await asyncio.gather(*tasks)
    return results
"""
        control = f"""def {fn}(workers: list, payload: dict) -> list:
    results = [worker.process(payload) for worker in workers]
    return results
"""
        pairs.append({
            "category": "async_await",
            "target_construct": "AsyncFunctionDef",
            "diff_summary": f"Async asyncio.gather vs synchronous list comprehension: {fn}",
            "target_code": target,
            "control_code": control,
        })

    return pairs[:60]


# ==============================================================================
# CATEGORY 4: CLASS_DEF (60 Pairs)
# ==============================================================================
def create_class_def_pairs() -> List[Dict[str, Any]]:
    pairs = []
    
    # Family 1: Numerical Accumulators & Metric Trackers (15 pairs)
    acc_specs = [
        ("RunningAverage", "total", "float", "count", "int", "def update(self, val: float): self.total += val; self.count += 1; return self.total / self.count", "make_running_average"),
        ("MovingVariance", "sum_x", "float", "sum_x2", "float", "def add(self, x: float): self.sum_x += x; self.sum_x2 += (x * x); return self.sum_x2", "make_moving_variance"),
        ("MinMaxTracker", "min_v", "float", "max_v", "float", "def observe(self, v: float): self.min_v = min(self.min_v, v); self.max_v = max(self.max_v, v); return (self.min_v, self.max_v)", "make_min_max_tracker"),
        ("LossAccumulator", "loss_sum", "float", "steps", "int", "def log_loss(self, l: float): self.loss_sum += l; self.steps += 1; return self.loss_sum / self.steps", "make_loss_accumulator"),
        ("GradientNormTracker", "norm_sum", "float", "n_batches", "int", "def step(self, gn: float): self.norm_sum += gn; self.n_batches += 1; return self.norm_sum / self.n_batches", "make_grad_norm_tracker"),
        ("AccuracyMeter", "correct", "int", "total", "int", "def record(self, is_corr: bool): self.correct += int(is_corr); self.total += 1; return self.correct / self.total", "make_accuracy_meter"),
        ("TokenCounter", "prompt_toks", "int", "gen_toks", "int", "def add_tokens(self, p: int, g: int): self.prompt_toks += p; self.gen_toks += g; return self.prompt_toks + self.gen_toks", "make_token_counter"),
        ("ThroughputMeter", "items", "int", "elapsed", "float", "def add_batch(self, n: int, s: float): self.items += n; self.elapsed += s; return self.items / self.elapsed", "make_throughput_meter"),
        ("PerplexityTracker", "cross_entropy_sum", "float", "tokens", "int", "def record(self, loss: float, count: int): self.cross_entropy_sum += loss; self.tokens += count; return self.cross_entropy_sum / self.tokens", "make_perplexity_tracker"),
        ("LearningRateSchedule", "initial_lr", "float", "step_num", "int", "def get_lr(self, decay: float): self.step_num += 1; return self.initial_lr * (decay ** self.step_num)", "make_lr_schedule"),
        ("EmaEstimator", "shadow", "float", "decay", "float", "def update(self, val: float): self.shadow = self.decay * self.shadow + (1 - self.decay) * val; return self.shadow", "make_ema_estimator"),
        ("TopKAccuracy", "top1", "int", "top5", "int", "def update(self, h1: bool, h5: bool): self.top1 += int(h1); self.top5 += int(h5); return (self.top1, self.top5)", "make_topk_accuracy"),
        ("KLDivAccumulator", "kl_total", "float", "eval_steps", "int", "def record(self, kl: float): self.kl_total += kl; self.eval_steps += 1; return self.kl_total / self.eval_steps", "make_kl_accumulator"),
        ("RewardTracker", "discounted_r", "float", "gamma", "float", "def step_reward(self, r: float): self.discounted_r = r + self.gamma * self.discounted_r; return self.discounted_r", "make_reward_tracker"),
        ("EntropyMeter", "entropy_sum", "float", "n_samples", "int", "def log_entropy(self, e: float): self.entropy_sum += e; self.n_samples += 1; return self.entropy_sum / self.n_samples", "make_entropy_meter"),
    ]
    for c_name, f1, t1, f2, t2, m_code, f_name in acc_specs:
        target = f"""class {c_name}:
    def __init__(self, {f1}: {t1} = 0, {f2}: {t2} = 0):
        self.{f1} = {f1}
        self.{f2} = {f2}

    {m_code}
"""
        control = f"""def {f_name}({f1}: {t1} = 0, {f2}: {t2} = 0) -> dict:
    state = {{'{f1}': {f1}, '{f2}': {f2}}}
    return state

def update_{f_name}(state: dict, val: float) -> float:
    state['{f1}'] += val
    state['{f2}'] += 1
    return state['{f1}'] / state['{f2}'] if state['{f2}'] != 0 else 0.0
"""
        pairs.append({
            "category": "class_def",
            "target_construct": "ClassDef",
            "diff_summary": f"Stateful class definition vs closure/dictionary state: {c_name}",
            "target_code": target,
            "control_code": control,
        })

    # Family 2: Data Structure Containers & Buffers (15 pairs)
    ds_specs = [
        ("RingBuffer", "capacity: int = 16", "self.buffer = []", "def push(self, item): self.buffer.append(item); return len(self.buffer)", "make_ring_buffer"),
        ("TokenStack", "max_depth: int = 100", "self.stack = []", "def push(self, tok: int): self.stack.append(tok); return len(self.stack)", "make_token_stack"),
        ("KeyValStore", "namespace: str = 'default'", "self.store = {}", "def set_key(self, k: str, v: str): self.store[k] = v; return v", "make_kv_store"),
        ("VectorList", "dim: int = 128", "self.vectors = []", "def add_vec(self, v: list): self.vectors.append(v); return len(self.vectors)", "make_vector_list"),
        ("AttentionCache", "n_layers: int = 32", "self.k_cache = {}", "def store_k(self, layer: int, k): self.k_cache[layer] = k; return layer", "make_attention_cache"),
        ("ActivationTape", "record_grads: bool = False", "self.tape = []", "def record_act(self, act): self.tape.append(act); return len(self.tape)", "make_activation_tape"),
        ("PriorityIndex", "size_limit: int = 50", "self.elements = []", "def insert_item(self, item, score: float): self.elements.append((score, item)); return len(self.elements)", "make_priority_index"),
        ("SymbolTable", "scope: str = 'global'", "self.symbols = {}", "def define(self, sym: str, val): self.symbols[sym] = val; return val", "make_symbol_table"),
        ("SlidingWindow", "window_size: int = 10", "self.window = []", "def append_val(self, v: float): self.window.append(v); return sum(self.window)", "make_sliding_window"),
        ("CheckpointRegistry", "base_path: str = './ckpts'", "self.entries = []", "def register(self, step: int): self.entries.append(step); return len(self.entries)", "make_checkpoint_registry"),
        ("BatchBuffer", "target_batch: int = 64", "self.samples = []", "def add_sample(self, s: dict): self.samples.append(s); return len(self.samples)", "make_batch_buffer"),
        ("PrefixTrie", "root_char: str = '^'", "self.nodes = {}", "def insert_prefix(self, p: str): self.nodes[p] = True; return len(self.nodes)", "make_prefix_trie"),
        ("NeuronMask", "n_neurons: int = 4096", "self.active = set()", "def activate(self, idx: int): self.active.add(idx); return len(self.active)", "make_neuron_mask"),
        ("LayerNormParams", "dim: int = 768", "self.gamma = [1.0] * dim", "def get_gamma(self): return len(self.gamma)", "make_layernorm_params"),
        ("GrammarRuleSet", "name: str = 'python'", "self.rules = []", "def add_rule(self, rule: str): self.rules.append(rule); return len(self.rules)", "make_grammar_rules"),
    ]
    for c_name, init_args, init_body, method_code, f_name in ds_specs:
        target = f"""class {c_name}:
    def __init__(self, {init_args}):
        {init_body}

    {method_code}
"""
        control = f"""def {f_name}({init_args}) -> dict:
    return {{'state': []}}

def {f_name}_push(store: dict, item) -> int:
    store['state'].append(item)
    return len(store['state'])
"""
        pairs.append({
            "category": "class_def",
            "target_construct": "ClassDef",
            "diff_summary": f"Data container class vs dictionary buffer: {c_name}",
            "target_code": target,
            "control_code": control,
        })

    # Family 3: Model Configuration & Hyperparameters (15 pairs)
    cfg_specs = [
        ("TransformerConfig", "d_model: int = 768, n_layers: int = 12, n_heads: int = 12"),
        ("GemmaMLPConfig", "hidden_dim: int = 3072, intermediate_dim: int = 8192, act_fn: str = 'gelu'"),
        ("AttentionConfig", "num_heads: int = 16, num_kv_heads: int = 4, head_dim: int = 64"),
        ("RoPEConfig", "base: float = 10000.0, scaling_factor: float = 1.0, max_seq_len: int = 8192"),
        ("OptimizerConfig", "learning_rate: float = 1e-4, weight_decay: float = 0.01, beta1: float = 0.9"),
        ("TrainingArguments", "batch_size: int = 32, max_steps: int = 10000, eval_interval: int = 500"),
        ("TokenizerConfig", "vocab_size: int = 32000, bos_id: int = 2, eos_id: int = 1"),
        ("LoraConfig", "r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05"),
        ("QuantizationConfig", "bits: int = 4, quant_type: str = 'nf4', use_double_quant: bool = True"),
        ("ProbeConfig", "probe_layer: int = 16, target_ast: str = 'Try', probe_type: str = 'linear'"),
        ("InterventionConfig", "ablate_heads: list = None, clamp_value: float = 0.0, layer_idx: int = 12"),
        ("DatasetConfig", "corpus_file: str = 'corpus.jsonl', max_tokens: int = 512, seed: int = 42"),
        ("GenerationConfig", "temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 128"),
        ("WandbLoggerConfig", "project: str = 'gemma_clean_code', entity: str = 'baddest-cmd', log_freq: int = 10"),
        ("EvaluatorConfig", "metric: str = 'monosemanticity_score', threshold: float = 0.85, n_runs: int = 5"),
    ]
    for idx, (c_name, params) in enumerate(cfg_specs, 1):
        target = f"""class {c_name}:
    def __init__(self, {params}):
        self.params = {{{params.split(',')[0].split(':')[0]}: 1}}

    def get_summary(self) -> dict:
        return self.params
"""
        control = f"""def build_{c_name.lower()}({params}) -> dict:
    return {{{params.split(',')[0].split(':')[0]}: 1}}

def get_{c_name.lower()}_summary(cfg: dict) -> dict:
    return cfg
"""
        pairs.append({
            "category": "class_def",
            "target_construct": "ClassDef",
            "diff_summary": f"Configuration class vs factory dictionary: {c_name}",
            "target_code": target,
            "control_code": control,
        })

    # Family 4: Algorithmic Operators & Pipeline Stages (15 pairs)
    for idx in range(1, 16):
        c_name = f"ASTPipelineProcessor_{idx:02d}"
        f_name = f"process_ast_pipeline_{idx:02d}"
        target = f"""class {c_name}:
    def __init__(self, stage_id: int = {idx}):
        self.stage_id = stage_id

    def forward(self, input_tensor: list) -> list:
        return [x * self.stage_id for x in input_tensor]
"""
        control = f"""def {f_name}(input_tensor: list, stage_id: int = {idx}) -> list:
    return [x * stage_id for x in input_tensor]
"""
        pairs.append({
            "category": "class_def",
            "target_construct": "ClassDef",
            "diff_summary": f"Pipeline processor class vs standalone pure function: {c_name}",
            "target_code": target,
            "control_code": control,
        })

    return pairs[:60]


# ==============================================================================
# CATEGORY 5: NESTED_LOOP (60 Pairs)
# ==============================================================================
def create_nested_loop_pairs() -> List[Dict[str, Any]]:
    pairs = []
    
    # Family 1: 2D Matrix & Tensor Grid Traversals (15 pairs)
    matrix_specs = [
        ("sum_matrix_2d", "rows: int, cols: int", "total += (r * cols + c)", "rows * cols", "total += idx"),
        ("fill_zero_grid", "height: int, width: int", "grid[h][w] = 0.0", "height * width", "grid[idx // width][idx % width] = 0.0"),
        ("count_matches_2d", "n: int, m: int", "if r == c: hits += 1", "n * m", "if (idx // m) == (idx % m): hits += 1"),
        ("multiply_elements", "dim_x: int, dim_y: int", "prod *= (1.0 + x * y)", "dim_x * dim_y", "prod *= (1.0 + (idx // dim_y) * (idx % dim_y))"),
        ("find_max_coordinate", "r_len: int, c_len: int", "if r + c > best: best = r + c", "r_len * c_len", "if (idx // c_len) + (idx % c_len) > best: best = (idx // c_len) + (idx % c_len)"),
        ("scale_submatrix", "rows: int, cols: int", "grid[i][j] *= factor", "rows * cols", "grid[idx // cols][idx % cols] *= factor"),
        ("zero_upper_triangle", "size: int, dummy: int = 0", "if j > i: grid[i][j] = 0.0", "size * size", "if (idx % size) > (idx // size): grid[idx // size][idx % size] = 0.0"),
        ("accumulate_outer_product", "n: int, m: int", "acc += vec_a[i] * vec_b[j]", "n * m", "acc += vec_a[idx // m] * vec_b[idx % m]"),
        ("euclidean_distance_matrix", "n_pts: int, dummy: int = 0", "dist[i][j] = abs(pts[i] - pts[j])", "n_pts * n_pts", "dist[idx // n_pts][idx % n_pts] = abs(pts[idx // n_pts] - pts[idx % n_pts])"),
        ("mask_diagonal_entries", "dim: int, dummy: int = 0", "if r == c: mask[r][c] = 1", "dim * dim", "if (idx // dim) == (idx % dim): mask[idx // dim][idx % dim] = 1"),
        ("sliding_kernel_dot", "h_len: int, w_len: int", "s += feat[r][c] * kernel[r][c]", "h_len * w_len", "s += feat[idx // w_len][idx % w_len] * kernel[idx // w_len][idx % w_len]"),
        ("threshold_matrix_values", "n_r: int, n_c: int", "if val > cutoff: count += 1", "n_r * n_c", "if val > cutoff: count += 1"),
        ("accumulate_patch_stats", "p_h: int, p_w: int", "total += patch[i][j]", "p_h * p_w", "total += patch[idx // p_w][idx % p_w]"),
        ("transpose_copy_buffer", "n: int, m: int", "out_buf[c][r] = in_buf[r][c]", "n * m", "out_buf[idx % m][idx // m] = in_buf[idx // m][idx % m]"),
        ("clamp_grid_bounds", "n: int, m: int", "grid[r][c] = max(0.0, grid[r][c])", "n * m", "grid[idx // m][idx % m] = max(0.0, grid[idx // m][idx % m])"),
    ]
    for fn, sig, nested_action, total_expr, flat_action in matrix_specs:
        target = f"""def {fn}({sig}) -> float:
    total = 0.0
    for r in range({sig.split(',')[0].split(':')[0]}):
        for c in range({sig.split(',')[1].split(':')[0]}):
            {nested_action}
    return total
"""
        control = f"""def {fn}({sig}) -> float:
    total = 0.0
    total_elements = {total_expr}
    for idx in range(total_elements):
        {flat_action}
    return total
"""
        pairs.append({
            "category": "nested_loop",
            "target_construct": "NestedLoop",
            "diff_summary": f"2D grid nested loops vs 1D flat index loop: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 2: Pairwise List & Token Comparisons (15 pairs)
    pair_specs = [
        ("pairwise_token_overlap", "tokens_a", "tokens_b", "if a == b: overlap += 1", "set_b = set(tokens_b)\n    for a in tokens_a:\n        if a in set_b: overlap += 1"),
        ("cartesian_string_concat", "words_a", "words_b", "pairs.append(a + '_' + b)", "pairs = [f'{a}_{b}' for a, b in zip(words_a, words_b)]"),
        ("cross_entropy_pair_sum", "dist_p", "dist_q", "acc += p_val * q_val", "acc = sum(p * q for p, q in zip(dist_p, dist_q))"),
        ("find_duplicate_pairs", "items_a", "items_b", "if x == y: dups.append(x)", "dups = list(set(items_a).intersection(items_b))"),
        ("all_pairs_cosine_score", "embeds_u", "embeds_v", "scores.append(u * v)", "scores = [u * v for u, v in zip(embeds_u, embeds_v)]"),
        ("ngram_cooccurrence", "context_window", "target_window", "if w1 == w2: cooc += 1", "set_w2 = set(target_window)\n    for w1 in context_window:\n        if w1 in set_w2: cooc += 1"),
        ("match_vocab_subsets", "src_vocab", "tgt_vocab", "if s_word == t_word: common.append(s_word)", "common = [w for w in src_vocab if w in tgt_vocab]"),
        ("bipartite_edge_weight", "nodes_u", "nodes_v", "total_w += edge_matrix.get((u, v), 0)", "total_w = sum(edge_matrix.get((u, v), 0) for u in nodes_u for v in [nodes_v[0]])"),
        ("attention_mask_alignment", "query_toks", "key_toks", "if q_pos >= k_pos: mask_count += 1", "mask_count = sum(1 for idx in range(len(query_toks)) if query_toks[idx] >= key_toks[idx])"),
        ("find_longest_common_token", "seq_1", "seq_2", "if t1 == t2: matched.append(t1)", "matched = [t for t in seq_1 if t in seq_2]"),
        ("edit_distance_char_scan", "chars_a", "chars_b", "if c1 != c2: diffs += 1", "diffs = sum(1 for c1, c2 in zip(chars_a, chars_b) if c1 != c2)"),
        ("compute_jaccard_pairs", "sample_a", "sample_b", "if a in b: inter += 1", "inter = len(set(sample_a) & set(sample_b))"),
        ("count_inversions", "arr_a", "arr_b", "if val_a > val_b: inv += 1", "inv = sum(1 for a, b in zip(arr_a, arr_b) if a > b)"),
        ("pair_wise_diff_norm", "vec_1", "vec_2", "total_diff += abs(v1 - v2)", "total_diff = sum(abs(v1 - v2) for v1, v2 in zip(vec_1, vec_2))"),
        ("compare_predictions", "preds", "targets", "if p == t: corr += 1", "corr = sum(1 for p, t in zip(preds, targets) if p == t)"),
    ]
    for fn, l1, l2, nested_stmt, flat_body in pair_specs:
        target = f"""def {fn}({l1}: list, {l2}: list) -> int:
    result = 0
    for a in {l1}:
        for b in {l2}:
            {nested_stmt}
    return result
"""
        control = f"""def {fn}({l1}: list, {l2}: list) -> int:
    result = 0
    {flat_body}
    return result
"""
        pairs.append({
            "category": "nested_loop",
            "target_construct": "NestedLoop",
            "diff_summary": f"Nested pairwise loops vs single loop or set intersection: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 3: Multi-layer Network & Attention Indexing (15 pairs)
    for idx in range(1, 16):
        fn = f"head_dimension_projection_{idx:02d}"
        target = f"""def {fn}(n_heads: int, head_dim: int) -> list:
    projections = []
    for h in range(n_heads):
        for d in range(head_dim):
            projections.append((h, d, h * head_dim + d))
    return projections
"""
        control = f"""def {fn}(n_heads: int, head_dim: int) -> list:
    total_dim = n_heads * head_dim
    projections = [(idx // head_dim, idx % head_dim, idx) for idx in range(total_dim)]
    return projections
"""
        pairs.append({
            "category": "nested_loop",
            "target_construct": "NestedLoop",
            "diff_summary": f"Nested head and dimension loop vs 1D product iteration: {fn}",
            "target_code": target,
            "control_code": control,
        })

    # Family 4: Nested While Loops & Multi-Level Counters (15 pairs)
    for idx in range(1, 16):
        fn = f"nested_while_stepper_{idx:02d}"
        target = f"""def {fn}(max_i: int, max_j: int) -> int:
    count = 0
    i = 0
    while i < max_i:
        j = 0
        while j < max_j:
            count += (i + j)
            j += 1
        i += 1
    return count
"""
        control = f"""def {fn}(max_i: int, max_j: int) -> int:
    count = 0
    idx = 0
    total = max_i * max_j
    while idx < total:
        i = idx // max_j
        j = idx % max_j
        count += (i + j)
        idx += 1
    return count
"""
        pairs.append({
            "category": "nested_loop",
            "target_construct": "NestedLoop",
            "diff_summary": f"Nested while loops vs single flat while loop: {fn}",
            "target_code": target,
            "control_code": control,
        })

    return pairs[:60]


# ==============================================================================
# NON-CONSTRUCT KEYWORD CONTROL INJECTION (LEXICAL VS. STRUCTURAL DISAMBIGUATION)
# ==============================================================================
def inject_non_construct_keyword_control(cat_name: str, control_code: str) -> str:
    """Inject target keyword strings in non-construct AST positions (identifiers, string literals, comments)."""
    if cat_name == "try_except":
        injected_lines = [
            "    try_count = 3",
            '    _kw_literal = "try"',
            "    # try again",
        ]
    elif cat_name == "with_statement":
        injected_lines = [
            "    with_retry = True",
            '    _kw_literal = "with"',
        ]
    elif cat_name == "class_def":
        injected_lines = [
            '    class_name = "User"',
            '    _kw_literal = "class"',
        ]
    elif cat_name == "async_await":
        injected_lines = [
            "    async_mode = False",
            '    _kw_literal = "async"',
        ]
    else:
        injected_lines = [
            "    for_each = True",
            '    _kw_literal = "for"',
        ]

    lines = control_code.splitlines()
    out_lines = []
    inserted = False
    for line in lines:
        out_lines.append(line)
        if not inserted and line.lstrip().startswith("def ") and line.rstrip().endswith(":"):
            out_lines.extend(injected_lines)
            inserted = True

    if not inserted:
        out_lines = [line.lstrip() for line in injected_lines] + out_lines

    return "\n".join(out_lines) + "\n"


# ==============================================================================
# MAIN CORPUS BUILDER & VALIDATOR
# ==============================================================================
def build_and_validate_corpus(output_path: Path) -> List[Dict[str, Any]]:
    """Assemble all 5 categories, inject non-construct keyword controls, validate AST parsing, and write JSONL."""
    category_generators = [
        ("try_except", create_try_except_pairs),
        ("with_statement", create_with_statement_pairs),
        ("async_await", create_async_await_pairs),
        ("class_def", create_class_def_pairs),
        ("nested_loop", create_nested_loop_pairs),
    ]

    all_pairs: List[Dict[str, Any]] = []

    for cat_name, gen_fn in category_generators:
        cat_pairs = gen_fn()
        if len(cat_pairs) != 60:
            raise ValueError(f"Category {cat_name} produced {len(cat_pairs)} pairs, expected exactly 60.")
        
        for idx, item in enumerate(cat_pairs, 1):
            pair_id = f"{cat_name}_{idx:03d}"
            item["pair_id"] = pair_id
            item["control_code"] = inject_non_construct_keyword_control(cat_name, item["control_code"])
            all_pairs.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    verified_records = []
    syntax_error_count = 0

    for item in all_pairs:
        # Validate target snippet with ast.parse()
        try:
            validate_python_code(item["target_code"], f"{item['pair_id']} (target)")
            target_valid = True
        except ValueError as err:
            print(err)
            syntax_error_count += 1
            target_valid = False

        # Validate control snippet with ast.parse()
        try:
            validate_python_code(item["control_code"], f"{item['pair_id']} (control)")
            control_valid = True
        except ValueError as err:
            print(err)
            syntax_error_count += 1
            control_valid = False

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
            "target_valid": target_valid,
            "control_valid": control_valid,
        }
        verified_records.append(record)

    if syntax_error_count > 0:
        raise RuntimeError(f"Encountered {syntax_error_count} syntax errors during AST parsing.")

    # Write formatted JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in verified_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return verified_records


def main():
    root_dir = Path(__file__).resolve().parent.parent
    corpus_file = root_dir / "corpus" / "gemma_ast_corpus.jsonl"

    print(f"Generating 60 contrastive snippet pairs per category at: {corpus_file}")
    records = build_and_validate_corpus(corpus_file)

    categories = set(r["category"] for r in records)
    total_pairs = len(records)
    total_snippets = total_pairs * 2

    print(f"\n✅ All {total_snippets} snippets ({total_pairs} pairs) successfully verified with ast.parse():")
    for cat in sorted(categories):
        count = sum(1 for r in records if r["category"] == cat)
        print(f"  - {cat:16s}: {count} pairs ({count * 2} snippets)")

    print(f"\nCorpus saved to: {corpus_file}")


if __name__ == "__main__":
    main()
