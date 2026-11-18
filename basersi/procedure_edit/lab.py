import json
import os
import time
import zlib

import torch
from PIL import Image

from basersi.engine.request import _stock_bounds, load_rgb, smart_resize, view_pixels
from basersi.engine.sampling import EFFORT

THINK_END = "</think>"


def view_image(path, min_tokens=None, max_tokens=None, row=None):
    image = load_rgb(path, row)
    min_tokens, max_tokens = _stock_bounds(min_tokens, max_tokens)
    if not min_tokens and not max_tokens:
        return image
    min_pixels, max_pixels = view_pixels(min_tokens, max_tokens)
    width, height = image.size
    new_h, new_w = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    return image.resize((new_w, new_h), Image.BICUBIC) if (new_h, new_w) != (height, width) else image


def row_crc(row_id):
    return zlib.crc32(row_id.encode())


def in_fold(row_id, fold, n_folds):
    return row_crc(row_id) % n_folds == fold


def window_samples(traces, row_id, gen_lo, gen_hi, prompt_hi):
    return sorted(
        int(k) for k, s in traces[row_id]["samples"].items()
        if gen_lo <= int(s["tokens"]) <= gen_hi and int(s.get("prompt_tokens") or 0) <= prompt_hi
        and s.get("reasoning") and s.get("finish") == "stop"
    )


def pick_sample(traces, row_id, gen_lo, gen_hi, prompt_hi):
    samples = window_samples(traces, row_id, gen_lo, gen_hi, prompt_hi)
    if not samples:
        return None
    return samples[row_crc(row_id) % len(samples)]


def load_traces(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_items(path):
    with open(path, encoding="utf-8") as f:
        return [(line.split()[0], int(line.split()[1])) for line in f if line.strip()]


def write_items(path, items):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(f"{r} {k}\n" for r, k in items))
    os.replace(tmp, path)
    return path


class StopForward(Exception):
    pass


class Context:
    pass


class Lab:
    def __init__(self, model_dir, traces, device, tail_positions, f32=False, base_drops_prefix=True, max_reason=0, memory_fraction=None):
        from transformers import AutoProcessor
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_modeling
        from torch.nn.utils.stateless import _reparametrize_module
        self.device, self.f32, self.tail_positions = device, f32, int(tail_positions)
        self.base_drops_prefix, self.max_reason = base_drops_prefix, max_reason
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer
        start = time.time()
        if memory_fraction:
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(memory_fraction)), device=torch.device(device))
        self.model = qwen_modeling.Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval().requires_grad_(False)
        self.language_model = self.model.model.language_model
        self.layers = self.language_model.layers
        self.n_layers = len(self.layers)
        types = [layer.block_type for layer in self.layers]
        self.kw_full_layer = types.index("full_attention")
        self.kw_linear_layer = next((i for i, t in enumerate(types) if t != "full_attention"), self.kw_full_layer)
        self.kw_layers = sorted({0, self.kw_linear_layer, self.kw_full_layer})
        self.hidden = self.model.config.text_config.hidden_size
        self.image_token = self.model.config.image_token_id
        self.end_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.think_end_token = self.tokenizer.convert_tokens_to_ids(THINK_END)
        self.traces = load_traces(traces) if isinstance(traces, str) else traces
        self.state = dict(cap={}, out={}, patch={}, kw={})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.model.model.visual.to(torch.float32)
        self.language_model.embed_tokens.register_forward_hook(lambda m, i, o: o.float() if self.f32 else None)

        def to_float(x):
            if torch.is_tensor(x):
                return x.float() if x.is_floating_point() else x
            if isinstance(x, (tuple, list)):
                return type(x)(to_float(v) for v in x)
            return x

        def wrap_float(module):
            original = module.forward

            def forward(*args, **kw):
                if not self.f32:
                    return original(*args, **kw)
                args = to_float(args)
                kw = {k: to_float(v) for k, v in kw.items()}
                with _reparametrize_module(module, {n: p.float() for n, p in module.named_parameters()}):
                    return original(*args, **kw)
            module.forward = forward
        for block in self.layers:
            wrap_float(block)
        wrap_float(self.language_model.norm)
        self._register_hooks()
        print(f"[lab] model on {device} in {time.time() - start:.0f}s, f32={f32}, {self.n_layers} layers", flush=True)

    def _register_hooks(self):
        state = self.state

        def output_hook(kind, layer):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                patch = state["patch"].get((kind, layer))
                if patch is not None:
                    out = patch(out)
                capture = state["cap"].get(kind)
                if capture is not None:
                    result = capture(layer, out)
                    if result is not None:
                        state["out"][(kind, layer)] = result
                if patch is not None:
                    return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
            return hook

        def input_hook(kind, layer):
            def hook(module, args):
                x = args[0]
                patch = state["patch"].get((kind, layer))
                if patch is not None:
                    x = patch(x)
                capture = state["cap"].get(kind)
                if capture is not None:
                    result = capture(layer, x)
                    if result is not None:
                        state["out"][(kind, layer)] = result
                if patch is not None:
                    return (x,) + tuple(args[1:])
            return hook

        def layer_kwargs_hook(layer):
            def hook(module, args, kwargs):
                if state.get("grab"):
                    state["kw"][layer] = (args, dict(kwargs))
                    if layer == self.kw_layers[-1]:
                        raise StopForward()
            return hook
        for index, layer in enumerate(self.layers):
            layer.mlp.register_forward_hook(output_hook("mlp", index))
            layer.mlp.down_proj.register_forward_pre_hook(input_hook("neu", index))
            layer.post_attention_layernorm.register_forward_hook(output_hook("mlpin", index))
            (layer.self_attn if layer.block_type == "full_attention" else layer.linear_attn).register_forward_hook(output_hook("mix", index))
            if index in self.kw_layers:
                layer.register_forward_pre_hook(layer_kwargs_hook(index), with_kwargs=True)

    def gen_ids(self, row_id, sample, max_reason=0):
        record = self.traces[row_id]["samples"][str(sample)]
        max_reason = max_reason or self.max_reason
        gen = self.tokenizer((record["reasoning"] or "") + "\n</think>\n\n" + (record["content"] or ""), add_special_tokens=False).input_ids
        if record.get("finish") == "stop":
            gen = gen + [self.end_token]
        think_end = gen.index(self.think_end_token) if self.think_end_token in gen else len(gen)
        if max_reason and think_end > max_reason:
            gen = gen[:max_reason]
            think_end = len(gen)
        return gen, think_end

    def prep(self, row_id, mode, gen):
        record = self.traces[row_id]
        settings = record.get("settings") or {}
        images = [view_image(p, settings.get("min_image_tokens"), settings.get("max_image_tokens"), dict(id=row_id)) for p in record["image_paths"]]
        drop_prefix = mode == "base" and self.base_drops_prefix
        prefix = [{"type": "text", "text": record["prefix"].rstrip() + "\n\n"}] if (record.get("prefix") and not drop_prefix) else []
        parts = record["prompt"].split("<image>")
        if len(parts) - 1 == len(images) and images:
            content = list(prefix)
            for j, text in enumerate(parts):
                if text.strip():
                    content.append({"type": "text", "text": text.strip()})
                if j < len(images):
                    content.append({"type": "image", "image": images[j]})
        else:
            content = [{"type": "image", "image": q} for q in images] + prefix + [{"type": "text", "text": record["prompt"]}]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt", reasoning_effort=EFFORT, enable_thinking=True)
        prompt_len = inputs["input_ids"].shape[1]
        ids = torch.cat([inputs["input_ids"][0], torch.tensor(gen, dtype=inputs["input_ids"].dtype)])[None].to(self.device)
        total_len = ids.shape[1]
        feed = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inputs["pixel_values"].to(self.device, torch.float32),
                    image_grid_thw=inputs["image_grid_thw"].to(self.device))
        if "mm_token_type_ids" in inputs:
            feed["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"][0], torch.zeros(total_len - prompt_len, dtype=inputs["mm_token_type_ids"].dtype)])[None].to(self.device)
        state = self.state
        state["grab"] = True
        state["kw"] = {}
        try:
            with torch.inference_mode():
                self.model.model(**feed, use_cache=False)
        except StopForward:
            pass
        state["grab"] = False
        ctx = Context()
        ctx.row_id, ctx.mode, ctx.P, ctx.T, ctx.ids = row_id, mode, prompt_len, total_len, ids[0]
        ctx.h0 = state["kw"][0][0][0]
        ctx.kw_linear, ctx.kw_full = state["kw"][self.kw_linear_layer][1], state["kw"][self.kw_full_layer][1]
        state["kw"] = {}
        ctx.is_image = (ctx.ids[:prompt_len] == self.image_token)
        return ctx

    @torch.inference_mode()
    def run(self, ctx, start=0, h=None, stop=None):
        h = ctx.h0 if h is None else h
        stop = self.n_layers if stop is None else stop
        for index in range(start, stop):
            layer = self.layers[index]
            kw = ctx.kw_full if layer.block_type == "full_attention" else ctx.kw_linear
            h = layer(h, **kw)
        return h

    @torch.inference_mode()
    def logp(self, h_final, chunk=256):
        out = []
        weight = self.model.lm_head.weight
        for i in range(0, h_final.shape[0], chunk):
            x = self.language_model.norm(h_final[i: i + chunk][None])[0]
            z = (x.float() @ weight.float().T) if self.f32 else self.model.lm_head(x).float()
            out.append(torch.log_softmax(z, -1))
        return torch.cat(out)

    def clear(self):
        self.state["cap"] = {}
        self.state["patch"] = {}
        self.state["out"] = {}

    def align(self, ctx_base, ctx_text):
        ids_base, ids_text = ctx_base.ids, ctx_text.ids
        inserted = ctx_text.T - ctx_base.T
        n = min(ctx_base.T, ctx_text.T)
        differ = (ids_base[:n] != ids_text[:n]).nonzero()
        start = int(differ[0]) if len(differ) else n
        assert torch.equal(ids_base[start:], ids_text[start + inserted:]), "templates differ by more than one inserted block"
        return start, inserted

    def classes(self, ctx):
        prompt_len, total_len = ctx.P, ctx.T
        positions = torch.arange(total_len, device=ctx.ids.device)
        cls = torch.full((total_len,), 2, device=ctx.ids.device)
        cls[:prompt_len][ctx.is_image] = 1
        cls[prompt_len - self.tail_positions: prompt_len] = 3
        cls[prompt_len:] = 4
        return {name: positions[cls == j] for j, name in enumerate(["sys", "img", "txt", "tail", "gen"])}
