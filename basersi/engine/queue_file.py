import json
import os


def append_record(path, rec):
    data = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            data = b"\n" + data
        done = os.write(fd, data)
        while done < len(data):
            done += os.write(fd, data[done:])
    finally:
        os.close(fd)


def parse_line(line):
    try:
        x = json.loads(line)
        return (x if isinstance(x, dict) else None), []
    except ValueError:
        pass
    found = []
    p = line.find(b'{"', 1)
    while p > 0:
        try:
            x = json.loads(line[p:])
            if isinstance(x, dict):
                found = [x]
                break
        except ValueError:
            pass
        p = line.find(b'{"', p + 1)
    return None, found


def cut_key(line):
    if line.startswith(b'{"key": "'):
        end = line.find(b'"', 9)
        if end > 9:
            return line[9:end].decode("utf-8", "replace")
    return None


def complete_lines(buf):
    end = buf.rfind(b"\n") + 1
    return buf[:end].split(b"\n")[:-1], end


def records(buf, on_bad=None):
    lines, end = complete_lines(buf)
    out = []
    for line in lines:
        if not line.strip():
            continue
        x, rescued = parse_line(line)
        if x is not None:
            out.append(x)
            continue
        if on_bad is not None:
            on_bad(line, cut_key(line), rescued)
        out.extend(rescued)
    return out, end


def read_records(path, on_bad=None):
    with open(path, "rb") as f:
        return records(f.read(), on_bad)[0]
