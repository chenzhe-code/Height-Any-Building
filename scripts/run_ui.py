#!/usr/bin/env python3
"""Start the HeightAnyBuilding interface on a loopback address."""

import argparse
import gzip
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
from secrets import token_urlsafe
import sys
from threading import Lock, Thread
from urllib.parse import parse_qs, quote, urlsplit

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.runtime import ui_jobs, ui_lod1, ui_preview


ASSETS = Path(__file__).with_name("ui_assets")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/lod1.js": ("lod1.js", "text/javascript; charset=utf-8"),
}


class Server(ThreadingHTTPServer):
    def __init__(self, port, jobs):
        super().__init__(("127.0.0.1", port), Handler)
        self.jobs = jobs
        self.hosts = {f"{name}:{self.server_port}" for name in ("127.0.0.1", "localhost")}
        self.earthdata_cookie_name = f"hab_edl_{self.server_port}"
        self.earthdata_sessions = {}
        self.earthdata_lock = Lock()


class Handler(BaseHTTPRequestHandler):
    def send_body(self, status, content_type, body, encoding=None, cookie=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if encoding:
            self.send_header("Content-Encoding", encoding)
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def json(self, payload, status=HTTPStatus.OK, compress=False, *, cookie=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        encoding = None
        if compress and "gzip" in self.headers.get("Accept-Encoding", "").replace(" ", "").split(","):
            body = gzip.compress(body, compresslevel=3)
            encoding = "gzip"
        self.send_body(status, "application/json; charset=utf-8", body, encoding, cookie)

    def earthdata_token(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        token = cookie.get(self.server.earthdata_cookie_name)
        return token.value if token else None

    def earthdata_values(self, token):
        with self.server.earthdata_lock:
            return self.server.earthdata_sessions.get(token, {})

    def earthdata_cookie(self, token):
        name = self.server.earthdata_cookie_name
        value = token if token else ""
        expiry = "" if token else "; Max-Age=0"
        return f"{name}={value}; Path=/api; HttpOnly; SameSite=Strict{expiry}"

    def dispatch(self, callback):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in self.server.hosts or (origin and origin != f"http://{host}"):
            self.json({"error": "请使用服务器显示的本地地址。"}, HTTPStatus.FORBIDDEN)
            return
        try:
            callback(urlsplit(self.path))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ui_jobs.BusyError as error:
            self.json({"error": str(error)}, HTTPStatus.CONFLICT)
        except FileNotFoundError as error:
            self.json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            self.json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_GET(self):
        self.dispatch(self.get)

    def do_POST(self):
        self.dispatch(self.post)

    def get(self, url):
        jobs = self.server.jobs
        query = parse_qs(url.query)
        if url.path in STATIC:
            filename, kind = STATIC[url.path]
            self.send_body(HTTPStatus.OK, kind, (ASSETS / filename).read_bytes())
        elif url.path == "/api/config":
            self.json({"models": list(ui_jobs.DEPTH_MODELS), "building_models": ui_jobs.BUILDING_MODELS,
                       "work_root": str(jobs.root)})
        elif url.path == "/api/earthdata":
            self.json({"saved": bool(self.earthdata_values(self.earthdata_token()))})
        elif url.path == "/api/files":
            self.json(ui_jobs.list_directory(query.get("path", [""])[0], jobs.data_roots,
                                             query.get("kind", ["image"])[0]))
        elif url.path == "/api/jobs":
            self.json(jobs.list())
        else:
            parts = url.path.strip("/").split("/")
            if parts[:2] != ["api", "jobs"] or len(parts) not in (3, 4):
                raise FileNotFoundError("找不到页面。")
            job_id = parts[2]
            if len(parts) == 3:
                self.json(jobs.detail(job_id))
            elif parts[3] == "download":
                self.download(job_id, query.get("artifact", [""])[0])
            elif parts[3] in {"preview", "lod1"}:
                self.preview(job_id, query.get("artifact", [""])[0], parts[3])
            else:
                raise FileNotFoundError("找不到页面。")

    def preview(self, job_id, selected, view):
        folder, job = self.server.jobs.locate(job_id)
        allowed = {item["id"] for item in ui_jobs.collect_outputs(folder, job)}
        if selected not in allowed or Path(selected).suffix.lower() not in ui_jobs.RASTERS:
            raise FileNotFoundError("找不到可预览的高度栅格。")
        output = folder / "output"
        height = self.result_path(output, selected)
        footprint = ui_jobs.footprint_artifact(selected)
        vector = self.result_path(output, footprint) if footprint in allowed else None
        if view == "lod1":
            if vector is None:
                raise FileNotFoundError("此高度成果还没有可用于 LOD1 的建筑 GeoJSON。")
            self.json(ui_lod1.render(vector), compress=True)
        else:
            self.json(ui_preview.render(height, vector))

    @staticmethod
    def result_path(output, selected):
        path = (output / selected).resolve(strict=True)
        if not path.is_relative_to(output) or not path.is_file():
            raise FileNotFoundError("找不到此任务的结果文件。")
        return path

    def download(self, job_id, selected):
        folder, job = self.server.jobs.locate(job_id)
        if not any(item["id"] == selected for item in ui_jobs.collect_outputs(folder, job)):
            raise FileNotFoundError("找不到此任务的结果文件。")
        path = self.result_path(folder / "output", selected)
        with path.open("rb") as source:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name, safe='')}")
            self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                shutil.copyfileobj(source, self.wfile, 256 * 1024)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def post(self, url):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("请求须使用 JSON。")
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 65_536:
            raise ValueError("请求大小无效。")
        values = json.loads(self.rfile.read(size))
        if not isinstance(values, dict):
            raise ValueError("请求格式无效。")
        jobs = self.server.jobs
        token = self.earthdata_token()
        if url.path == "/api/earthdata/clear":
            with self.server.earthdata_lock:
                self.server.earthdata_sessions.pop(token, None)
            self.json({"saved": False}, cookie=self.earthdata_cookie(None))
            return

        submitted = {field: values[field] for field in ui_jobs.EARTHDATA_FIELDS.values() if field in values}
        if not submitted:
            values.update(self.earthdata_values(token))

        if url.path == "/api/jobs":
            job, worker = jobs.start(values)
        else:
            parts = url.path.strip("/").split("/")
            if parts[:2] != ["api", "jobs"] or len(parts) != 4:
                raise FileNotFoundError("找不到操作。")
            _, _, job_id, action = parts
            if action == "cancel":
                jobs.cancel(job_id)
                self.json({"id": job_id}, HTTPStatus.ACCEPTED)
                return
            if action != "resume":
                raise FileNotFoundError("找不到操作。")
            job, worker = jobs.resume(job_id, values)
        Thread(target=worker.wait, daemon=True).start()
        cookie = None
        if submitted and ui_jobs.requires_earthdata(job):
            new_token = token_urlsafe(32)
            with self.server.earthdata_lock:
                self.server.earthdata_sessions.pop(token, None)
                self.server.earthdata_sessions[new_token] = submitted
            token = new_token
            cookie = self.earthdata_cookie(token)
        self.json({"id": job["id"], "earthdata_saved": bool(self.earthdata_values(token))},
                  HTTPStatus.ACCEPTED, cookie=cookie)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        ui_jobs.run_worker(Path(sys.argv[2]), int(sys.argv[3]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-root", type=Path, action="append", default=[],
                        help="Expose a local directory in the input browser; repeat for more directories")
    parser.add_argument("--work-root", type=Path, default=ui_jobs.ROOT / "work_dir/ui")
    args = parser.parse_args()
    roots = tuple(dict.fromkeys(path.expanduser().resolve() for path in (args.data_root or [Path.cwd()])))
    for root in roots:
        if not root.is_dir():
            parser.error(f"Data root is not a directory: {root}")
    try:
        jobs = ui_jobs.Jobs(args.work_root, roots)
        with Server(args.port, jobs) as server:
            print(f"HeightAnyBuilding UI: http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
