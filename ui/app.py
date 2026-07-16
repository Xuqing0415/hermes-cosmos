#!/usr/bin/env python3
"""
Hermes Cosmos Web Interface - 


 + 
"""

import random
import streamlit as st
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# ====================  ====================

# 
COLORS = {
    "bg_primary": "#0A0C10",
    "bg_card": "#11151A",
    "border": "#22262C",
    "text_primary": "#E6EDF3",
    "text_secondary": "#8A94A0",
    "accent_blue": "#2B9AFF",
    "accent_green": "#00D4AA",
    "accent_purple": "#B580FF",
    "warning": "#FF5A5A",
}

# 
FONT_STYLES = {
    "title": "font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 500; line-height: 1.3;",
    "heading": "font-family: 'Inter', system-ui, sans-serif; font-size: 20px; font-weight: 500; line-height: 1.4;",
    "body": "font-family: 'Inter', system-ui, sans-serif; font-size: 14px; font-weight: 400; line-height: 1.5;",
    "code": "font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 400; line-height: 1.4;",
    "badge": "font-family: 'Inter', system-ui, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.2;",
}

# CSS
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --bg-primary: {COLORS['bg_primary']};
    --bg-card: {COLORS['bg_card']};
    --border: {COLORS['border']};
    --text-primary: {COLORS['text_primary']};
    --text-secondary: {COLORS['text_secondary']};
    --accent-blue: {COLORS['accent_blue']};
    --accent-green: {COLORS['accent_green']};
    --accent-purple: {COLORS['accent_purple']};
}}

body {{
    background: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
    font-family: 'Inter', system-ui, sans-serif;
    line-height: 1.5;
}}

.card {{
    background: {COLORS['bg_card']};
    border-radius: 12px;
    padding: 20px;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    border: 1px solid {COLORS['border']};
}}

.card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    border: 1px solid rgba(43,154,255,0.3);
}}

.progress-bar {{
    height: 4px;
    background: #2A2F36;
    border-radius: 2px;
    overflow: hidden;
}}

.progress-fill {{
    background: linear-gradient(90deg, {COLORS['accent_blue']}, {COLORS['accent_green']});
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
}}

.terminal {{
    background: rgba(0,0,0,0.5);
    backdrop-filter: blur(4px);
    font-family: 'JetBrains Mono', monospace;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid {COLORS['border']};
}}

.emoji-avatar {{
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    background: rgba(43,154,255,0.1);
    border: 1px solid rgba(43,154,255,0.3);
}}

.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}}

.status-blue {{ background: {COLORS['accent_blue']}; }}
.status-green {{ background: {COLORS['accent_green']}; }}
.status-purple {{ background: {COLORS['accent_purple']}; }}
.status-red {{ background: {COLORS['warning']}; }}
</style>
"""

# ====================  ====================

def generate_fitness_data(points: int = 50) -> List[float]:
    """"""
    data = [50.0]
    for _ in range(points - 1):
        change = random.uniform(-5, 10)
        new_val = max(0, min(100, data[-1] + change))
        data.append(new_val)
    return data

def generate_events(count: int = 20) -> List[Dict[str, Any]]:
    """"""
    roles = ["", "", "", ""]
    role_colors = {"": "blue", "": "purple", "": "green", "": "red"}
    
    messages = {
        "": [
            "",
            "",
            "",
            "",
            "",
        ],
        "": [
            " {}  {}%",
            "",
            "",
            "",
            "",
        ],
        "": [
            "",
            "",
            "",
            "...",
            "",
        ],
        "": [
            "",
            "",
            "",
            "",
            "",
        ],
    }
    
    events = []
    for i in range(count):
        role = random.choice(roles)
        msg = random.choice(messages[role])
        if "{}" in msg:
            msg = msg.format(random.randint(1, 100), random.randint(60, 90))
        
        events.append({
            "id": i,
            "role": role,
            "color": role_colors[role],
            "message": msg,
            "timestamp": datetime.now() - timedelta(minutes=i * random.randint(5, 30)),
        })
    
    return events

def generate_characters(count: int = 6) -> List[Dict[str, Any]]:
    """"""
    races = ["", "", "", "", ""]
    classes = ["", "", "", "", ""]
    
    characters = []
    for i in range(count):
        mood = random.randint(0, 100)
        mood_color = "#00D4AA" if mood > 60 else "#FFB800" if mood > 30 else "#FF5A5A"
        
        characters.append({
            "id": i,
            "name": f"{i+1}",
            "race": random.choice(races),
            "class": random.choice(classes),
            "level": random.randint(1, 20),
            "experience": random.randint(0, 100),
            "mood": mood,
            "mood_color": mood_color,
            "abilities": random.sample(["", "", "", "", ""], 3),
            "avatar": ["", "", "", "", ""][i % 5],
        })
    
    return characters

def generate_timeline() -> List[Dict[str, Any]]:
    """"""
    timeline = [
        {"generation": "1-50", "title": "", "description": "", "ascii": "  .  \n *** \n*****"},
        {"generation": "51-100", "title": "", "description": "", "ascii": "  *  \n *** \n*****"},
        {"generation": "101-150", "title": "", "description": "50%", "ascii": " *** \n*****\n*****"},
        {"generation": "151-200", "title": "", "description": "", "ascii": "*****\n*****\n*****"},
        {"generation": "201-250", "title": "", "description": "", "ascii": " *****\n*******\n *****"},
    ]
    return timeline

def generate_friends_feed(count: int = 8) -> List[Dict[str, Any]]:
    """"""
    users = [
        {"name": "", "avatar": "", "role": ""},
        {"name": "", "avatar": "", "role": ""},
        {"name": "", "avatar": "", "role": ""},
        {"name": "", "avatar": "", "role": ""},
        {"name": "", "avatar": "", "role": "NPC"},
    ]
    
    posts = []
    for i in range(count):
        user = random.choice(users)
        content = random.choice([
            f" {random.randint(60, 95)}%",
            f"... ",
            f" {random.randint(100, 500)}  ",
            f"",
            f"... ",
        ])
        
        posts.append({
            "id": i,
            "user": user,
            "content": content,
            "time": f"{random.randint(1, 24)}",
            "likes": random.randint(0, 50),
            "comments": [
                {"user": random.choice(users)["name"], "text": random.choice(["", "", "", ""])}
                for _ in range(random.randint(0, 3))
            ],
        })
    
    return posts

# ====================  ====================

def render_sidebar():
    """"""
    st.sidebar.markdown(
        f"<h1 style='{FONT_STYLES['title']} background: linear-gradient(135deg, {COLORS['accent_blue']}, {COLORS['accent_green']}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'> Hermes</h1>",
        unsafe_allow_html=True
    )
    
    pages = [
        (" ", "dashboard"),
        (" ", "documentary"),
        (" ", "friends"),
        (" ", "characters"),
        (" ", "console"),
    ]
    
    selected_page = st.sidebar.radio("", [p[0] for p in pages])
    return dict(pages)[selected_page]

def render_metric_card(title: str, value: str, subtitle: str = "", trend: float = None):
    """"""
    with st.container():
        st.markdown(f"<div class='card'><h3 style='{FONT_STYLES['heading']}'>{title}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 32px; font-weight: 500; color: {COLORS['accent_blue']};'>{value}</p>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>{subtitle}</p>", unsafe_allow_html=True)
        if trend is not None:
            trend_color = COLORS['accent_green'] if trend > 0 else COLORS['warning']
            trend_icon = "↑" if trend > 0 else "↓"
            st.markdown(f"<span style='color: {trend_color};'>{trend_icon} {abs(trend)}%</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_fitness_chart():
    """"""
    data = generate_fitness_data(50)
    
    # 
    max_val = max(data)
    chart_lines = []
    for i in range(len(data) - 1, -1, -5):
        height = int(data[i] / 10)
        line = "" * height + "" * (10 - height)
        chart_lines.append(f"{i:3d} | {line}")
    
    chart_lines.reverse()
    
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='{FONT_STYLES['heading']}'> </h3>", unsafe_allow_html=True)
    st.markdown(f"<pre style='{FONT_STYLES['code']} color: {COLORS['accent_blue']};'>{'\n'.join(chart_lines)}</pre>", unsafe_allow_html=True)
    st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'> 50 </p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_event_log(events: List[Dict[str, Any]]):
    """"""
    st.markdown(f"<div class='terminal'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='{FONT_STYLES['heading']}'> </h3>", unsafe_allow_html=True)
    
    for event in events[:10]:
        color_class = f"status-{event['color']}"
        timestamp = event['timestamp'].strftime("%H:%M:%S")
        st.markdown(
            f"<span class='status-dot {color_class}'></span><span style='{FONT_STYLES['code']}'>[{timestamp}] [{event['role']}] {event['message']}</span>",
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_character_card(character: Dict[str, Any]):
    """"""
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    
    # 
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<div class='emoji-avatar'>{character['avatar']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h4 style='{FONT_STYLES['heading']}'>{character['name']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>{character['race']} · {character['class']}</p>", unsafe_allow_html=True)
    
    # 
    st.markdown(f"<div style='margin: 12px 0;'><span style='{FONT_STYLES['badge']}'></span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='progress-bar'><div class='progress-fill' style='width: {character['mood']}%; background: {character['mood_color']};'></div></div>", unsafe_allow_html=True)
    
    # 
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"<span style='{FONT_STYLES['badge']}'> {character['level']}</span>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<span style='{FONT_STYLES['badge']}'> {character['experience']}%</span>", unsafe_allow_html=True)
    
    # 
    st.markdown(f"<div style='margin-top: 12px;'>", unsafe_allow_html=True)
    for ability in character['abilities']:
        st.markdown(f"<span style='{FONT_STYLES['badge']} background: rgba(138,148,160,0.2); padding: 4px 8px; border-radius: 4px; margin-right: 4px;'>{ability}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 
    st.markdown(f"<div style='margin-top: 16px;'>", unsafe_allow_html=True)
    st.button(" ", key=f"like_{character['id']}", help="")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_timeline(timeline: List[Dict[str, Any]]):
    """"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'> </h2>", unsafe_allow_html=True)
    
    for item in timeline:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 12px;'>", unsafe_allow_html=True)
        st.markdown(f"<span style='{FONT_STYLES['badge']} background: rgba(43,154,255,0.2); padding: 4px 12px; border-radius: 12px;'> {item['generation']}</span>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='{FONT_STYLES['heading']}'>{item['title']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>{item['description']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<pre style='{FONT_STYLES['code']} background: #0D1117; padding: 12px; border-radius: 8px; color: {COLORS['accent_green']};'>{item['ascii']}</pre>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_friends_feed(posts: List[Dict[str, Any]]):
    """"""
    for post in posts:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        
        # 
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f"<div class='emoji-avatar'>{post['user']['avatar']}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h4 style='{FONT_STYLES['heading']}'>{post['user']['name']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>{post['time']}</p>", unsafe_allow_html=True)
        
        # 
        st.markdown(f"<p style='{FONT_STYLES['body']}; margin: 12px 0;'>{post['content']}</p>", unsafe_allow_html=True)
        
        # 
        st.markdown(f"<div style='display: flex; gap: 16px;'>", unsafe_allow_html=True)
        st.markdown(f"<span style='{FONT_STYLES['badge']}'> {post['likes']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='{FONT_STYLES['badge']}'> {len(post['comments'])}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 
        if post['comments']:
            st.markdown(f"<div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid {COLORS['border']};'>", unsafe_allow_html=True)
            for comment in post['comments']:
                st.markdown(f"<p style='{FONT_STYLES['code']} color: {COLORS['text_secondary']};'><strong>{comment['user']}</strong>: {comment['text']}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_console():
    """"""
    st.markdown(f"<div class='terminal'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='{FONT_STYLES['heading']}'> </h3>", unsafe_allow_html=True)
    
    # 
    if "console_history" not in st.session_state:
        st.session_state.console_history = [
            {"type": "system", "text": " Hermes  'help' "},
        ]
    
    for entry in st.session_state.console_history:
        if entry["type"] == "system":
            color = COLORS['text_secondary']
        elif entry["type"] == "command":
            color = COLORS['accent_blue']
        elif entry["type"] == "success":
            color = COLORS['accent_green']
        else:
            color = COLORS['warning']
        
        st.markdown(f"<span style='{FONT_STYLES['code']} color: {color};'>{entry['text']}</span>", unsafe_allow_html=True)
    
    # 
    command = st.text_input("", value="", placeholder="> ...", key="console_input")
    
    if command:
        st.session_state.console_history.append({"type": "command", "text": f"> {command}"})
        
        # 
        if command == "help":
            help_text = """:
  status        - 
  boost [attr] [val]  -  (: boost fitness +10)
  spawn event   - 
  time warp [n] -  n 
  reset         - 
  help          - """
            st.session_state.console_history.append({"type": "success", "text": help_text})
        elif command.startswith("boost"):
            st.session_state.console_history.append({"type": "success", "text": " "})
        elif command == "spawn event":
            events = ["", "", "", ""]
            st.session_state.console_history.append({"type": "success", "text": f" : {random.choice(events)}"})
        elif command.startswith("time warp"):
            st.session_state.console_history.append({"type": "success", "text": "⏰ "})
        elif command == "reset":
            st.session_state.console_history.append({"type": "success", "text": " "})
        elif command == "status":
            st.session_state.console_history.append({"type": "success", "text": f""" :
  : {random.randint(60, 90)}%
  : {random.randint(100, 500)}
  : {random.randint(3, 10)}""".strip()})
        else:
            st.session_state.console_history.append({"type": "error", "text": f" : {command}"})
        
        # 
        st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ====================  ====================

def render_dashboard():
    """"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'> </h2>", unsafe_allow_html=True)
    
    # 
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("", "87.3%", "", trend=2.5)
    with col2:
        render_metric_card("", "342", "", trend=None)
    with col3:
        render_metric_card("", "6", "", trend=1)
    
    # 
    col4, col5 = st.columns([2, 1])
    with col4:
        render_fitness_chart()
    with col5:
        events = generate_events(20)
        render_event_log(events)

def render_documentary():
    """"""
    timeline = generate_timeline()
    render_timeline(timeline)

def render_friends():
    """"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'> </h2>", unsafe_allow_html=True)
    posts = generate_friends_feed(8)
    render_friends_feed(posts)

def render_characters():
    """"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'> </h2>", unsafe_allow_html=True)
    
    characters = generate_characters(6)
    col1, col2 = st.columns(2)
    
    for i, char in enumerate(characters):
        if i % 2 == 0:
            with col1:
                render_character_card(char)
        else:
            with col2:
                render_character_card(char)

# ====================  ====================

def main():
    """"""
    st.set_page_config(page_title="Hermes Cosmos", layout="wide")
    
    # 
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # 
    selected_page = render_sidebar()
    
    # 
    with st.container():
        if selected_page == "dashboard":
            render_dashboard()
        elif selected_page == "documentary":
            render_documentary()
        elif selected_page == "friends":
            render_friends()
        elif selected_page == "characters":
            render_characters()
        elif selected_page == "console":
            st.markdown(f"<h2 style='{FONT_STYLES['title']}'> </h2>", unsafe_allow_html=True)
            render_console()

if __name__ == "__main__":
    main()
