import pandas as pd
import os
from datetime import datetime

class AttendanceManager:
    def __init__(self, file_path='attendance_log.csv'):
        self.file_path = file_path
        self.columns = ['Name', 'ID', 'Date', 'Time', 'Status']
        # In-memory cache of today's marked IDs to avoid redundant CSV reads
        self._marked_today = set()
        self._ensure_file_exists()
        self._load_todays_cache()

    def _ensure_file_exists(self):
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.file_path, index=False)

    def _load_todays_cache(self):
        """Pre-load today's marked IDs to avoid reading CSV on every frame."""
        today = datetime.now().strftime('%Y-%m-%d')
        try:
            df = pd.read_csv(self.file_path, dtype={'ID': str})
            if not df.empty and 'Date' in df.columns:
                todays = df[df['Date'] == today]
                self._marked_today = set(todays['ID'].astype(str).unique())
        except Exception:
            self._marked_today = set()

    def mark_attendance(self, name, student_id):
        """
        Mark attendance for a student.
        Returns (True, message) if marked successfully, (False, message) if already marked today.
        Uses an in-memory set for O(1) duplicate checks per session.
        """
        sid_str = str(student_id)

        if sid_str in self._marked_today:
            return False, "Already Marked"

        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        # Double-check against CSV for cross-session correctness
        try:
            df = pd.read_csv(self.file_path, dtype={'ID': str})
            if not df.empty:
                existing = df[(df['ID'].astype(str) == sid_str) & (df['Date'] == date_str)]
                if not existing.empty:
                    self._marked_today.add(sid_str)
                    return False, "Already Marked"
        except Exception:
            df = pd.DataFrame(columns=self.columns)

        # Append new entry
        new_entry = pd.DataFrame(
            [[name, sid_str, date_str, time_str, 'Present']],
            columns=self.columns
        )
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(self.file_path, index=False)

        self._marked_today.add(sid_str)
        return True, f"✓ Marked Present"

    def get_attendance_log(self):
        """Return the full attendance log as a DataFrame."""
        try:
            df = pd.read_csv(self.file_path, dtype={'ID': str})
            if not df.empty:
                if 'Status' not in df.columns:
                    df['Status'] = 'Present'
                else:
                    df['Status'] = df['Status'].fillna('Present')
            return df
        except Exception:
            return pd.DataFrame(columns=self.columns)

    def get_todays_attendance(self):
        """Return today's attendance records."""
        today = datetime.now().strftime('%Y-%m-%d')
        df = self.get_attendance_log()
        if df.empty or 'Date' not in df.columns:
            return pd.DataFrame(columns=self.columns)
        return df[df['Date'] == today]

    def get_summary(self):
        """Return a summary dict with aggregate stats."""
        df = self.get_attendance_log()
        today = datetime.now().strftime('%Y-%m-%d')
        today_df = df[df['Date'] == today] if not df.empty else pd.DataFrame()
        return {
            'total_records': len(df),
            'today_count': len(today_df),
            'unique_students_today': today_df['ID'].nunique() if not today_df.empty else 0,
            'all_dates': sorted(df['Date'].unique().tolist(), reverse=True) if not df.empty else [],
        }
