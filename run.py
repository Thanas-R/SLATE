"""
run.py
Single-command launcher for SLATE: Spectral Analysis of Neural Network Dynamics.
Initializes the mathematical pipeline, starts the local server, and launches your browser.
No emojis. No em dashes.
"""

import sys
import os
import webbrowser
import threading
import time

def print_ascii_banner():
    print(r"""
+--------------------------------------------------------------------+
|  SLATE :: SPECTRAL ANALYSIS OF NEURAL NETWORK DYNAMICS          |
|  Mathematical Research & Discrete Dynamical Systems Telemetry      |
+--------------------------------------------------------------------+
    """)

def open_browser(url, delay=1.2):
    time.sleep(delay)
    print(f"[*] Opening browser: {url}")
    webbrowser.open(url)

def main():
    print_ascii_banner()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        print("[*] Launching Streamlit dashboard...")
        os.system("streamlit run streamlit_app.py")
        return

    port = 8000
    url = f"http://127.0.0.1:{port}"
    
    print("[*] Initializing pipeline...")
    from core.pipeline import SLATEPipeline
    pipeline = SLATEPipeline(hidden_dim=8)
    results = pipeline.run_experiment()
    
    print(f"[+] Pipeline initialized:")
    print(f"    - Timesteps T:       {results['meta']['timesteps_T']}")
    print(f"    - Latent States:     R^{results['meta']['hidden_dim_n']}")
    print(f"    - Spectral Radius:   {results['spectral']['spectral_radius']} ({results['spectral']['stability_class']})")
    print(f"    - Linear Fit R^2:    {results['dynamics']['r2_fit_score']}")
    print(f"[+] Server starting at: {url}")
    print("[+] Press Ctrl+C in terminal to terminate.")
    
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    import uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    main()
