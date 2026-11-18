"""
Income Classification & Gender-Fairness Audit Web Server
University of Delhi - Department of Computer Science
Candidate: Ekta Sachdeva | Roll Number: 24020570063 | Problem 24
"""

import http.server
import socketserver
import webbrowser
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print("="*70)
        print("INCOME CLASSIFICATION & GENDER-FAIRNESS AUDIT WEB DASHBOARD")
        print("Department of Computer Science - University of Delhi")
        print("Student: Ekta Sachdeva | Roll No: 24020570063")
        print(f"Serving at: {url}")
        print("="*70)
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == '__main__':
    main()
