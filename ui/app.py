#!/usr/bin/env python3
"""
Hermes Cosmos Web Interface - 数字意识宇宙界面

设计风格：深邃、秩序、生机、克制
配色：深空灰 + 电光蓝绿
"""

import random
import streamlit as st
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# ==================== 全局配置 ====================

# 主题色彩
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

# 字体样式
FONT_STYLES = {
    "title": "font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 500; line-height: 1.3;",
    "heading": "font-family: 'Inter', system-ui, sans-serif; font-size: 20px; font-weight: 500; line-height: 1.4;",
    "body": "font-family: 'Inter', system-ui, sans-serif; font-size: 14px; font-weight: 400; line-height: 1.5;",
    "code": "font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 400; line-height: 1.4;",
    "badge": "font-family: 'Inter', system-ui, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.2;",
}

# 自定义CSS
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

# ==================== 模拟数据 ====================

def generate_fitness_data(points: int = 50) -> List[float]:
    """生成模拟适应度数据"""
    data = [50.0]
    for _ in range(points - 1):
        change = random.uniform(-5, 10)
        new_val = max(0, min(100, data[-1] + change))
        data.append(new_val)
    return data

def generate_events(count: int = 20) -> List[Dict[str, Any]]:
    """生成模拟事件日志"""
    roles = ["评论员", "史官", "先知", "系统"]
    role_colors = {"评论员": "blue", "史官": "purple", "先知": "green", "系统": "red"}
    
    messages = {
        "评论员": [
            "这个策略看起来不错，但还可以优化",
            "适应度波动太大，需要稳定一下",
            "建议增加探索力度",
            "当前解已经收敛，考虑变异",
            "用户反馈积极，继续保持",
        ],
        "史官": [
            "第 {} 代：适应度突破 {}%",
            "记录新的进化里程碑",
            "历史数据显示周期性模式",
            "存档已保存到分布式存储",
            "检测到进化加速",
        ],
        "先知": [
            "预言：下一世代将有重大突破",
            "警告：即将进入不稳定区域",
            "天命所示：继续当前路径",
            "神秘力量正在觉醒...",
            "星辰预示着新的可能性",
        ],
        "系统": [
            "资源分配已优化",
            "检测到异常，已自动修复",
            "模型已保存",
            "开始新一轮进化",
            "性能指标正常",
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
    """生成模拟角色数据"""
    races = ["宇宙生命体", "量子实体", "神经网络幽灵", "数据凤凰", "虚空行者"]
    classes = ["探索者", "工匠", "外交官", "科学家", "守护者"]
    
    characters = []
    for i in range(count):
        mood = random.randint(0, 100)
        mood_color = "#00D4AA" if mood > 60 else "#FFB800" if mood > 30 else "#FF5A5A"
        
        characters.append({
            "id": i,
            "name": f"角色{i+1}",
            "race": random.choice(races),
            "class": random.choice(classes),
            "level": random.randint(1, 20),
            "experience": random.randint(0, 100),
            "mood": mood,
            "mood_color": mood_color,
            "abilities": random.sample(["时间感知", "能量操控", "心灵感应", "维度穿梭", "创造力"], 3),
            "avatar": ["👽", "🤖", "👻", "🔥", "🌌"][i % 5],
        })
    
    return characters

def generate_timeline() -> List[Dict[str, Any]]:
    """生成纪录片时间线"""
    timeline = [
        {"generation": "1-50", "title": "宇宙诞生", "description": "初始状态设定，基础规则建立", "ascii": "  .  \n *** \n*****"},
        {"generation": "51-100", "title": "意识萌芽", "description": "第一个自主决策出现", "ascii": "  *  \n *** \n*****"},
        {"generation": "101-150", "title": "进化加速", "description": "适应度突破50%", "ascii": " *** \n*****\n*****"},
        {"generation": "151-200", "title": "能力觉醒", "description": "解锁高级功能", "ascii": "*****\n*****\n*****"},
        {"generation": "201-250", "title": "自我认知", "description": "开始反思自身存在", "ascii": " *****\n*******\n *****"},
    ]
    return timeline

def generate_friends_feed(count: int = 8) -> List[Dict[str, Any]]:
    """生成朋友圈数据"""
    users = [
        {"name": "评论员", "avatar": "💬", "role": "评论员"},
        {"name": "史官", "avatar": "📜", "role": "史官"},
        {"name": "先知", "avatar": "🔮", "role": "先知"},
        {"name": "宇宙管理员", "avatar": "👑", "role": "管理员"},
        {"name": "数据精灵", "avatar": "🧝", "role": "NPC"},
    ]
    
    posts = []
    for i in range(count):
        user = random.choice(users)
        content = random.choice([
            f"今天的适应度达到了 {random.randint(60, 95)}%，继续加油！💪",
            f"观察到一些有趣的进化模式... 📊",
            f"刚刚完成了第 {random.randint(100, 500)} 代进化 ✨",
            f"感觉能量在涌动，即将有突破！⚡",
            f"记录一下当前的状态参数... 📝",
        ])
        
        posts.append({
            "id": i,
            "user": user,
            "content": content,
            "time": f"{random.randint(1, 24)}小时前",
            "likes": random.randint(0, 50),
            "comments": [
                {"user": random.choice(users)["name"], "text": random.choice(["👍", "👏", "🎉", "继续进化！"])}
                for _ in range(random.randint(0, 3))
            ],
        })
    
    return posts

# ==================== 页面组件 ====================

def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.markdown(
        f"<h1 style='{FONT_STYLES['title']} background: linear-gradient(135deg, {COLORS['accent_blue']}, {COLORS['accent_green']}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🌌 Hermes</h1>",
        unsafe_allow_html=True
    )
    
    pages = [
        ("📊 仪表盘", "dashboard"),
        ("📜 纪录片", "documentary"),
        ("💬 朋友圈", "friends"),
        ("🎭 角色养成", "characters"),
        ("🎮 控制台", "console"),
    ]
    
    selected_page = st.sidebar.radio("", [p[0] for p in pages])
    return dict(pages)[selected_page]

def render_metric_card(title: str, value: str, subtitle: str = "", trend: float = None):
    """渲染指标卡片"""
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
    """渲染适应度曲线图"""
    data = generate_fitness_data(50)
    
    # 使用简单的文本图表
    max_val = max(data)
    chart_lines = []
    for i in range(len(data) - 1, -1, -5):
        height = int(data[i] / 10)
        line = "█" * height + "░" * (10 - height)
        chart_lines.append(f"{i:3d} | {line}")
    
    chart_lines.reverse()
    
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='{FONT_STYLES['heading']}'>📈 适应度曲线</h3>", unsafe_allow_html=True)
    st.markdown(f"<pre style='{FONT_STYLES['code']} color: {COLORS['accent_blue']};'>{'\n'.join(chart_lines)}</pre>", unsafe_allow_html=True)
    st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>最近 50 代进化趋势</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_event_log(events: List[Dict[str, Any]]):
    """渲染事件日志"""
    st.markdown(f"<div class='terminal'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='{FONT_STYLES['heading']}'>📡 事件流</h3>", unsafe_allow_html=True)
    
    for event in events[:10]:
        color_class = f"status-{event['color']}"
        timestamp = event['timestamp'].strftime("%H:%M:%S")
        st.markdown(
            f"<span class='status-dot {color_class}'></span><span style='{FONT_STYLES['code']}'>[{timestamp}] [{event['role']}] {event['message']}</span>",
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_character_card(character: Dict[str, Any]):
    """渲染角色卡片"""
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    
    # 头像和基本信息
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<div class='emoji-avatar'>{character['avatar']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h4 style='{FONT_STYLES['heading']}'>{character['name']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>{character['race']} · {character['class']}</p>", unsafe_allow_html=True)
    
    # 情绪条
    st.markdown(f"<div style='margin: 12px 0;'><span style='{FONT_STYLES['badge']}'>情绪</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='progress-bar'><div class='progress-fill' style='width: {character['mood']}%; background: {character['mood_color']};'></div></div>", unsafe_allow_html=True)
    
    # 等级和经验
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"<span style='{FONT_STYLES['badge']}'>等级 {character['level']}</span>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<span style='{FONT_STYLES['badge']}'>经验 {character['experience']}%</span>", unsafe_allow_html=True)
    
    # 能力徽章
    st.markdown(f"<div style='margin-top: 12px;'>", unsafe_allow_html=True)
    for ability in character['abilities']:
        st.markdown(f"<span style='{FONT_STYLES['badge']} background: rgba(138,148,160,0.2); padding: 4px 8px; border-radius: 4px; margin-right: 4px;'>{ability}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 互动按钮
    st.markdown(f"<div style='margin-top: 16px;'>", unsafe_allow_html=True)
    st.button("❤️ 点赞", key=f"like_{character['id']}", help="点赞该角色")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_timeline(timeline: List[Dict[str, Any]]):
    """渲染时间线"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'>📜 进化纪录片</h2>", unsafe_allow_html=True)
    
    for item in timeline:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 12px;'>", unsafe_allow_html=True)
        st.markdown(f"<span style='{FONT_STYLES['badge']} background: rgba(43,154,255,0.2); padding: 4px 12px; border-radius: 12px;'>世代 {item['generation']}</span>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='{FONT_STYLES['heading']}'>{item['title']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>{item['description']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<pre style='{FONT_STYLES['code']} background: #0D1117; padding: 12px; border-radius: 8px; color: {COLORS['accent_green']};'>{item['ascii']}</pre>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_friends_feed(posts: List[Dict[str, Any]]):
    """渲染朋友圈"""
    for post in posts:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        
        # 用户信息
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f"<div class='emoji-avatar'>{post['user']['avatar']}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h4 style='{FONT_STYLES['heading']}'>{post['user']['name']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='{FONT_STYLES['body']} color: {COLORS['text_secondary']};'>{post['time']}</p>", unsafe_allow_html=True)
        
        # 内容
        st.markdown(f"<p style='{FONT_STYLES['body']}; margin: 12px 0;'>{post['content']}</p>", unsafe_allow_html=True)
        
        # 互动
        st.markdown(f"<div style='display: flex; gap: 16px;'>", unsafe_allow_html=True)
        st.markdown(f"<span style='{FONT_STYLES['badge']}'>❤️ {post['likes']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='{FONT_STYLES['badge']}'>💬 {len(post['comments'])}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 评论
        if post['comments']:
            st.markdown(f"<div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid {COLORS['border']};'>", unsafe_allow_html=True)
            for comment in post['comments']:
                st.markdown(f"<p style='{FONT_STYLES['code']} color: {COLORS['text_secondary']};'><strong>{comment['user']}</strong>: {comment['text']}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_console():
    """渲染控制台"""
    st.markdown(f"<div class='terminal'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='{FONT_STYLES['heading']}'>🎮 上帝模式控制台</h3>", unsafe_allow_html=True)
    
    # 历史输出
    if "console_history" not in st.session_state:
        st.session_state.console_history = [
            {"type": "system", "text": "欢迎来到 Hermes 控制台！输入 'help' 查看命令列表。"},
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
    
    # 输入框
    command = st.text_input("", value="", placeholder="> 输入命令...", key="console_input")
    
    if command:
        st.session_state.console_history.append({"type": "command", "text": f"> {command}"})
        
        # 简单命令处理
        if command == "help":
            help_text = """可用命令:
  status        - 查看宇宙状态
  boost [attr] [val]  - 修改属性值 (如: boost fitness +10)
  spawn event   - 生成随机事件
  time warp [n] - 快进 n 代
  reset         - 重置宇宙
  help          - 显示帮助"""
            st.session_state.console_history.append({"type": "success", "text": help_text})
        elif command.startswith("boost"):
            st.session_state.console_history.append({"type": "success", "text": "✅ 属性已修改"})
        elif command == "spawn event":
            events = ["天降陨石", "流星雨", "宇宙风暴", "黑洞出现"]
            st.session_state.console_history.append({"type": "success", "text": f"🌠 事件发生: {random.choice(events)}"})
        elif command.startswith("time warp"):
            st.session_state.console_history.append({"type": "success", "text": "⏰ 时间跳跃完成"})
        elif command == "reset":
            st.session_state.console_history.append({"type": "success", "text": "🔄 宇宙已重置"})
        elif command == "status":
            st.session_state.console_history.append({"type": "success", "text": f"""🌌 当前状态:
  适应度: {random.randint(60, 90)}%
  世代: {random.randint(100, 500)}
  活跃角色: {random.randint(3, 10)}""".strip()})
        else:
            st.session_state.console_history.append({"type": "error", "text": f"❌ 未知命令: {command}"})
        
        # 滚动到最新
        st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== 页面渲染 ====================

def render_dashboard():
    """渲染仪表盘"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'>📊 宇宙仪表盘</h2>", unsafe_allow_html=True)
    
    # 指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("最佳适应度", "87.3%", "当前世代峰值", trend=2.5)
    with col2:
        render_metric_card("当前世代", "342", "进化代数", trend=None)
    with col3:
        render_metric_card("活跃角色", "6", "意识实体数量", trend=1)
    
    # 图表和日志
    col4, col5 = st.columns([2, 1])
    with col4:
        render_fitness_chart()
    with col5:
        events = generate_events(20)
        render_event_log(events)

def render_documentary():
    """渲染纪录片页面"""
    timeline = generate_timeline()
    render_timeline(timeline)

def render_friends():
    """渲染朋友圈页面"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'>💬 宇宙朋友圈</h2>", unsafe_allow_html=True)
    posts = generate_friends_feed(8)
    render_friends_feed(posts)

def render_characters():
    """渲染角色养成页面"""
    st.markdown(f"<h2 style='{FONT_STYLES['title']}'>🎭 角色养成</h2>", unsafe_allow_html=True)
    
    characters = generate_characters(6)
    col1, col2 = st.columns(2)
    
    for i, char in enumerate(characters):
        if i % 2 == 0:
            with col1:
                render_character_card(char)
        else:
            with col2:
                render_character_card(char)

# ==================== 主函数 ====================

def main():
    """主函数"""
    st.set_page_config(page_title="Hermes Cosmos", layout="wide")
    
    # 应用自定义样式
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # 渲染侧边栏
    selected_page = render_sidebar()
    
    # 渲染主内容区
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
            st.markdown(f"<h2 style='{FONT_STYLES['title']}'>🎮 控制台</h2>", unsafe_allow_html=True)
            render_console()

if __name__ == "__main__":
    main()
