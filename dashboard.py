"""
Smart Attendance System – Web Dashboard
Run: python dashboard.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, jsonify, request, redirect, url_for
from attendance_manager import AttendanceManager
from datetime import datetime

app = Flask(__name__, template_folder='templates')

ATTENDANCE_FILE = os.path.join(os.path.dirname(__file__), 'attendance_log.csv')
mgr = AttendanceManager(file_path=ATTENDANCE_FILE)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    summary = mgr.get_summary()
    return render_template('index.html', summary=summary)


@app.route('/api/attendance')
def api_attendance():
    """Return all attendance records as JSON (optionally filtered by date)."""
    date_filter = request.args.get('date', '')
    df = mgr.get_attendance_log()
    if date_filter and not df.empty:
        df = df[df['Date'] == date_filter]
    return jsonify({
        'records': df.to_dict(orient='records'),
        'count': len(df),
    })


@app.route('/api/summary')
def api_summary():
    return jsonify(mgr.get_summary())


@app.route('/api/dates')
def api_dates():
    summary = mgr.get_summary()
    return jsonify({'dates': summary['all_dates']})


@app.route('/api/add', methods=['POST'])
def api_add_manual():
    """Manually mark attendance via the dashboard."""
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    sid  = data.get('id', '').strip()
    if not name or not sid:
        return jsonify({'success': False, 'message': 'Name and ID are required.'}), 400
    success, msg = mgr.mark_attendance(name, sid)
    return jsonify({'success': success, 'message': msg})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
