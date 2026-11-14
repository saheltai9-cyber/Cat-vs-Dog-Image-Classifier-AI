"""
Automated Demo Snapshot Capture via Chrome DevTools Protocol (CDP).
Captures genuine, live application screenshots with real model inference.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import websockets

PROJECT_DIR = Path(__file__).resolve().parent
SNAPSHOTS_DIR = PROJECT_DIR / "demo_snapshots" / "project05"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CAT_TEST_IMAGE = PROJECT_DIR / "data" / "raw" / "cats_and_dogs_filtered" / "validation" / "cats" / "cat.2000.jpg"
DOG_TEST_IMAGE = PROJECT_DIR / "data" / "raw" / "cats_and_dogs_filtered" / "validation" / "dogs" / "dog.2000.jpg"

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA_DIR = r"c:\temp\chrome_cdp_profile"


class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self._msg_id = 0

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=50 * 1024 * 1024)

    async def send(self, method, params=None):
        self._msg_id += 1
        msg_id = self._msg_id
        payload = {"id": msg_id, "method": method}
        if params:
            payload["params"] = params
        await self.ws.send(json.dumps(payload))
        while True:
            resp = json.loads(await self.ws.recv())
            if resp.get("id") == msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP error in {method}: {resp['error']}")
                return resp.get("result", {})

    async def evaluate(self, expression):
        res = await self.send("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True
        })
        return res.get("result", {}).get("value")

    async def wait_for_selector(self, selector, timeout=10.0, interval=0.3):
        start = time.time()
        while time.time() - start < timeout:
            val = await self.evaluate(f"!!document.querySelector('{selector}')")
            if val:
                return True
            await asyncio.sleep(interval)
        return False

    async def capture_screenshot(self, output_path: Path):
        res = await self.send("Page.captureScreenshot", {"format": "png"})
        img_bytes = base64.b64decode(res["data"])
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        print(f"[+] Saved screenshot: {output_path.name} ({len(img_bytes)} bytes)")

    async def close(self):
        if self.ws:
            await self.ws.close()


async def main():
    print("[*] Launching headless Chrome with remote debugging on port 9222...")
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        f"--user-data-dir={USER_DATA_DIR}",
        "--window-size=1280,900"
    ])
    time.sleep(2.5)

    try:
        # Get WebSocket debugger URL for target
        with urllib.request.urlopen("http://127.0.0.1:9222/json") as resp:
            targets = json.loads(resp.read().decode())
        ws_url = targets[0]["webSocketDebuggerUrl"]
        print(f"[+] Connected to Chrome target: {ws_url}")

        client = CDPClient(ws_url)
        await client.connect()
        await client.send("Page.enable")
        await client.send("DOM.enable")
        await client.send("Runtime.enable")

        # Set viewport
        await client.send("Emulation.setDeviceMetricsOverride", {
            "width": 1280,
            "height": 900,
            "deviceScaleFactor": 1,
            "mobile": False
        })

        # ====================================================================
        # 1. Main Dashboard
        # ====================================================================
        print("\n[*] Navigating to Main Dashboard (http://127.0.0.1:8080/)...")
        await client.send("Page.navigate", {"url": "http://127.0.0.1:8080/"})
        await client.wait_for_selector(".pv-metric-card", timeout=8.0)
        await asyncio.sleep(2.0)  # Wait for NiceGUI Vue render & transitions
        await client.capture_screenshot(SNAPSHOTS_DIR / "01_main_dashboard.png")

        # ====================================================================
        # 2. Classifier Page (Awaiting Upload)
        # ====================================================================
        print("\n[*] Navigating to Classifier Page (http://127.0.0.1:8080/classifier)...")
        await client.send("Page.navigate", {"url": "http://127.0.0.1:8080/classifier"})
        await client.wait_for_selector("input[type='file']", timeout=8.0)
        await asyncio.sleep(1.5)
        await client.capture_screenshot(SNAPSHOTS_DIR / "02_classifier_page.png")

        # ====================================================================
        # 3. Cat Image Classification
        # ====================================================================
        print(f"\n[*] Uploading test cat image: {CAT_TEST_IMAGE.name}...")
        doc = await client.send("DOM.getDocument")
        file_input = await client.send("DOM.querySelector", {
            "nodeId": doc["root"]["nodeId"],
            "selector": "input[type='file']"
        })
        await client.send("DOM.setFileInputFiles", {
            "files": [str(CAT_TEST_IMAGE)],
            "nodeId": file_input["nodeId"]
        })
        print("[*] Waiting for upload processing and preview to appear...")
        # Wait for "Classify Image" button to appear
        classify_btn_ready = await client.wait_for_selector("button:has(.q-icon):not([disabled])", timeout=8.0)
        await asyncio.sleep(2.0)

        # Click Classify Image
        print("[*] Clicking 'Classify Image' button to run real model inference...")
        await client.evaluate("""
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const btn = btns.find(b => b.innerText.includes('Classify Image'));
                if (btn) btn.click();
            })()
        """)

        # Wait for prediction result card
        print("[*] Waiting for neural network inference result...")
        await client.wait_for_selector(".pv-badge-cat, .pv-badge-dog", timeout=10.0)
        await asyncio.sleep(2.0)
        await client.capture_screenshot(SNAPSHOTS_DIR / "03_cat_classification.png")

        # Also capture focused prediction result card as 04_prediction_result.png
        await client.capture_screenshot(SNAPSHOTS_DIR / "04_prediction_result.png")

        # ====================================================================
        # 4. Dog Image Classification
        # ====================================================================
        print(f"\n[*] Clicking 'Clear' and uploading test dog image: {DOG_TEST_IMAGE.name}...")
        await client.evaluate("""
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const btn = btns.find(b => b.innerText.includes('Clear'));
                if (btn) btn.click();
            })()
        """)
        await asyncio.sleep(1.5)

        # Refresh doc and find file input again
        doc = await client.send("DOM.getDocument")
        file_input = await client.send("DOM.querySelector", {
            "nodeId": doc["root"]["nodeId"],
            "selector": "input[type='file']"
        })
        await client.send("DOM.setFileInputFiles", {
            "files": [str(DOG_TEST_IMAGE)],
            "nodeId": file_input["nodeId"]
        })
        await asyncio.sleep(2.0)

        print("[*] Clicking 'Classify Image' for dog image...")
        await client.evaluate("""
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const btn = btns.find(b => b.innerText.includes('Classify Image'));
                if (btn) btn.click();
            })()
        """)
        await client.wait_for_selector(".pv-badge-dog", timeout=10.0)
        await asyncio.sleep(2.0)
        # Save dog classification
        await client.capture_screenshot(SNAPSHOTS_DIR / "04_dog_classification.png")

        # ====================================================================
        # 5. Model Insights Page
        # ====================================================================
        print("\n[*] Navigating to Model Insights (http://127.0.0.1:8080/insights)...")
        await client.send("Page.navigate", {"url": "http://127.0.0.1:8080/insights"})
        await client.wait_for_selector(".js-plotly-plot", timeout=10.0)
        await asyncio.sleep(2.5)  # Let Plotly charts render completely
        await client.capture_screenshot(SNAPSHOTS_DIR / "05_model_insights.png")

        await client.close()
        print("\n[+] ALL 5 DEMO SNAPSHOTS CAPTURED SUCCESSFULLY!")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())

