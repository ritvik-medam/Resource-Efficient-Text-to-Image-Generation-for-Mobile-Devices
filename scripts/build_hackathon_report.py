"""
Hackathon Technical Solution Report Builder
Converts docs/HACKATHON_TECHNICAL_REPORT_TEMPLATE.html into publication-quality PDF via Chrome Headless.
"""

import os
import base64
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
TEMPLATE_PATH = os.path.join(DOCS_DIR, "HACKATHON_TECHNICAL_REPORT_TEMPLATE.html")
HTML_OUT = os.path.join(DOCS_DIR, "HACKATHON_TECHNICAL_REPORT.html")
PDF_OUT = os.path.join(DOCS_DIR, "HACKATHON_TECHNICAL_REPORT.pdf")
ROOT_PDF_OUT = os.path.join(BASE_DIR, "HACKATHON_TECHNICAL_REPORT.pdf")

def get_base64_img(filename):
    path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: Image {path} not found")
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def build_hackathon_report():
    print("Encoding images...")
    panda_b64 = get_base64_img("demo_sample_panda.png")
    v0_b64 = get_base64_img("verification_0.png")
    arch_b64 = get_base64_img("system_architecture.png")

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    print("Replacing placeholders...")
    content = content.replace("{{PANDA_B64}}", panda_b64)
    content = content.replace("{{V0_B64}}", v0_b64)
    content = content.replace("{{ARCH_B64}}", arch_b64)

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved populated HTML: {HTML_OUT}")

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--user-data-dir=/tmp/chrome_hackathon_pdf_profile",
        f"--print-to-pdf={PDF_OUT}",
        "--no-pdf-header-footer",
        f"file://{HTML_OUT}"
    ]

    print("Rendering Hackathon Technical Solution Report PDF via Chrome...")
    res = subprocess.run(chrome_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Chrome error:", res.stderr)
        raise RuntimeError("PDF compilation failed")

    # Copy to root directory
    shutil.copyfile(PDF_OUT, ROOT_PDF_OUT)
    size_kb = os.path.getsize(PDF_OUT) / 1024
    print(f"Successfully generated Hackathon Technical Solution Report PDF: {PDF_OUT} ({size_kb:.1f} KB)")
    print(f"Copied root PDF: {ROOT_PDF_OUT}")

if __name__ == "__main__":
    build_hackathon_report()
