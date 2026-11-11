import re

# NOTE: This migration script is kept for reference.
# The latest styles and icons are already applied directly to app.py.
# sec_opts now uses :material/ icons for a clean line-icon style.

file_path = 'app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# ── Icon migration: ensure sec_opts uses material icons ──────────────────────
old_sec_emoji = '''        sec_opts = [
            "🎯 AI Lead Intelligence",
            "🔍 Patient Insight Card",
        ]'''
old_sec_unicode = '''        sec_opts = [
            "⌖ AI Lead Intelligence",
            "⌕ Patient Insight Card",
        ]'''
new_sec = '''        sec_opts = [
            ":material/my_location: AI Lead Intelligence",
            ":material/search: Patient Insight Card",
        ]'''
text = text.replace(old_sec_emoji, new_sec)
text = text.replace(old_sec_unicode, new_sec)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated styles and icons!')
