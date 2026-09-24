"""Headless screenshots of the web viewer for paper figures.

Serves viewer/ locally, routes the three.js CDN URLs to a local copy (npm pack
three@0.128.0), and captures views per case with software WebGL.

  python scripts/render_viewer.py --three /path/to/three/package --case lung_004 --out paper/fig
"""
import argparse
import functools
import http.server
import os
import threading

from playwright.sync_api import sync_playwright

CDN = {
    "three.min.js": "build/three.min.js",
    "OrbitControls.js": "examples/js/controls/OrbitControls.js",
}


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(directory):
    h = functools.partial(_Quiet, directory=directory)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--three", required=True, help="unpacked three@0.128.0 npm package dir")
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--out", default="paper/fig")
    ap.add_argument("--views", default="A,tumor")
    ap.add_argument("--hide", default="", help="comma list of structure keys to hide")
    ap.add_argument("--opacity", type=float, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srv = serve(os.path.abspath("viewer"))
    url = f"http://127.0.0.1:{srv.server_port}/index.html"
    exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe if os.path.exists(exe) else None,
                              args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = b.new_page(viewport={"width": 1500, "height": 950}, device_scale_factor=1.5)
        def handle(route):
            u = route.request.url
            for name, rel in CDN.items():
                if u.endswith(name):
                    return route.fulfill(path=os.path.join(a.three, rel), content_type="application/javascript")
            if u.startswith("http://127.0.0.1"):
                return route.continue_()
            return route.abort()  # no external network (fonts fall back to system faces)
        page.route("**/*", handle)
        page.on("pageerror", lambda e: print("pageerror:", e))
        page.goto(url)
        for case in a.case:
            page.wait_for_function("document.querySelector('#case option')", timeout=60000)
            if page.input_value("#case") != case:
                page.select_option("#case", case)
            page.wait_for_function(f"document.getElementById('file').textContent.includes('{case}') && document.getElementById('loading').hidden", timeout=180000)
            for key in [k for k in a.hide.split(",") if k]:
                page.evaluate(f"(() => {{ const e = document.getElementById('s-{key}'); if (e && e.checked) e.click(); }})()")
            if a.opacity is not None:
                page.evaluate(f"(() => {{ const e = document.getElementById('op'); e.value = {a.opacity}; e.dispatchEvent(new Event('input')); }})()")
            for v in a.views.split(","):
                page.click(f"[data-view='{v}']")
                page.wait_for_timeout(1500)
                page.locator("#vp").screenshot(path=os.path.join(a.out, f"{case}_{v}{a.tag}.png"))
                print("saved", case, v)
            page.screenshot(path=os.path.join(a.out, f"{case}_ui{a.tag}.png"))
        b.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
