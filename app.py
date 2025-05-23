from flask import Flask, render_template_string
import subprocess
import threading
import os
import sys

app = Flask(__name__)

# HTML template to embed Streamlit app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head></head>
<header>
    <title>AI-Solutions Sales Dashboard</title>
    <style>
        iframe {
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            height: 60%;
            border: none;
        }
       @media only screen and (max-width: 768px) {
        .block-container {
            padding: 0.5rem !important;
        } 
    </style>
</header>
<body>
    <iframe src="http://localhost:8501/" frameborder="0"></iframe>
</body>
</html>
"""

def run_streamlit():
    """Start the Streamlit app as a subprocess."""
    streamlit_script = os.path.join(os.getcwd(), "1_Dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_script])

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    threading.Thread(target=run_streamlit).start()
    app.run(debug=True, port=8500)
