import streamlit as st
import requests
import time
import json
from datetime import datetime

API_GATEWAY = "http://localhost:8000"

# 页面配置
st.set_page_config(
    page_title="Hermes AI训练调度系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
    <style>
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
    }
    .status-running {
        color: #10B981;
    }
    .status-pending {
        color: #F59E0B;
    }
    .status-recovered {
        color: #3B82F6;
    }
    .status-failed {
        color: #EF4444;
    }
    </style>
    """, unsafe_allow_html=True)

# 主标题
st.title("🚀 Project Hermes - AI训练调度系统")
st.markdown("---")

# 侧边栏：提交作业
with st.sidebar:
    st.header("📝 提交新作业")
    
    job_id = st.text_input("作业ID", value=f"job-{int(time.time())}")
    model_name = st.selectbox("模型名称", ["tiny_model", "resnet50", "bert-base", "gpt2-small", "llama-7b"])
    num_gpus = st.slider("GPU数量", 1, 8, 1)
    batch_size = st.number_input("Batch Size", 8, 128, 32)
    epochs = st.number_input("训练轮数", 1, 10, 2)
    checkpoint_interval = st.number_input("Checkpoint间隔(步)", 10, 1000, 100)
    priority = st.selectbox("优先级", ["low", "normal", "high", "critical"])
    
    if st.button("🚀 提交作业", type="primary"):
        with st.spinner("提交中..."):
            payload = {
                "job_id": job_id,
                "model_name": model_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "gpu_type": "A100",
                "num_gpus": num_gpus,
                "checkpoint_interval": checkpoint_interval,
                "priority": priority
            }
            try:
                resp = requests.post(f"{API_GATEWAY}/api/v1/jobs", json=payload, timeout=5)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"✅ 作业 **{job_id}** 提交成功！")
                    st.info(f"📌 调度到: {result.get('region', 'Unknown')}")
                else:
                    st.error(f"❌ 提交失败: {resp.text}")
            except Exception as e:
                st.error(f"❌ 连接失败: {str(e)}")

# 主区域
col1, col2 = st.columns(2)

# 左侧：作业状态
with col1:
    st.header("📊 作业状态监控")
    
    job_id_query = st.text_input("查询作业ID", value="real-test-001", key="query")
    
    if st.button("🔍 查询状态"):
        try:
            resp = requests.get(f"{API_GATEWAY}/api/v1/jobs/{job_id_query}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                
                # 状态卡片
                status = data.get("status", "unknown").upper()
                region = data.get("region", "Unknown")
                gpu_count = data.get("gpu_count", 0)
                
                # 状态颜色
                status_color = {
                    "RUNNING": "#10B981",
                    "PENDING": "#F59E0B",
                    "RECOVERED": "#3B82F6",
                    "FAILED": "#EF4444"
                }.get(status, "#6B7280")
                
                # 展示卡片
                st.markdown(f"""
                <div style="background-color: #1F2937; padding: 20px; border-radius: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="color: white; margin: 0;">{data.get('name', job_id_query)}</h3>
                            <p style="color: #9CA3AF; margin: 5px 0 0 0;">Job ID: {job_id_query}</p>
                        </div>
                        <div style="background-color: {status_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold;">
                            {status}
                        </div>
                    </div>
                    <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr;">
                        <div style="color: #9CA3AF;">
                            <p style="margin: 0; color: #D1D5DB;">区域</p>
                            <p style="margin: 5px 0 0 0; color: white; font-size: 18px;">{region}</p>
                        </div>
                        <div style="color: #9CA3AF;">
                            <p style="margin: 0; color: #D1D5DB;">GPU数量</p>
                            <p style="margin: 5px 0 0 0; color: white; font-size: 18px;">{gpu_count}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 完整JSON
                with st.expander("查看完整信息"):
                    st.json(data)
            else:
                st.error("❌ 作业不存在或查询失败")
        except Exception as e:
            st.error(f"❌ 连接失败: {str(e)}")

    # 列出所有作业
    if st.button("📋 列出所有作业"):
        try:
            resp = requests.get(f"{API_GATEWAY}/api/v1/jobs", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.write(f"总共 {data.get('total', 0)} 个作业")
                for job in data.get('jobs', []):
                    st.write(f"- **{job.get('name')}** ({job.get('status')})")
        except Exception as e:
            st.error(f"❌ 连接失败: {str(e)}")

# 右侧：故障模拟
with col2:
    st.header("💥 故障模拟")
    
    st.markdown("""
    **模拟真实场景：**
    - 点击下方按钮模拟Pod被删除
    - 系统会自动检测故障并重新调度
    - 展示恢复时间
    """)
    
    if st.button("💥 模拟Pod删除 (故障注入)", type="primary"):
        if not job_id_query.strip():
            st.error("请先输入作业ID")
        else:
            with st.spinner("正在注入故障..."):
                try:
                    start_time = time.time()
                    resp = requests.post(f"{API_GATEWAY}/api/v1/jobs/{job_id_query}/fail", timeout=10)
                    end_time = time.time()
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        recovery_time = result.get('details', {}).get('recovery_time_ms', (end_time - start_time) * 1000)
                        
                        # 成功卡片
                        st.markdown(f"""
                        <div style="background-color: #059669; padding: 20px; border-radius: 12px;">
                            <div style="display: flex; align-items: center; gap: 15px;">
                                <span style="font-size: 40px;">✅</span>
                                <div>
                                    <h3 style="color: white; margin: 0;">故障恢复成功！</h3>
                                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">作业 {job_id_query} 已自动恢复</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        | 指标 | 值 |
                        |------|------|
                        | 原区域 | {result.get('details', {}).get('previous_region', 'N/A')} |
                        | 新区域 | {result.get('details', {}).get('new_region', 'N/A')} |
                        | 恢复时间 | **{recovery_time:.0f} ms** |
                        """)
                    else:
                        st.error(f"❌ 故障模拟失败: {resp.text}")
                except Exception as e:
                    st.error(f"❌ 连接失败: {str(e)}")

# 底部：集群状态
st.markdown("---")
st.header("🖥️ 集群状态")

if st.button("🔄 刷新集群状态"):
    try:
        resp = requests.get(f"{API_GATEWAY}/api/v1/cluster", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            
            total_gpus = data.get('total_gpus', 0)
            available_gpus = data.get('available_gpus', 0)
            used_gpus = total_gpus - available_gpus
            
            # 进度条
            st.progress(used_gpus / total_gpus if total_gpus > 0 else 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总GPU", total_gpus)
            with col2:
                st.metric("已使用", used_gpus)
            with col3:
                st.metric("可用", available_gpus)
            
            # 区域详情
            st.subheader("区域分布")
            regions = data.get('regions', {})
            for region, info in regions.items():
                st.write(f"- **{region}**: {info.get('available_gpus', 0)}/{info.get('total_gpus', 0)} GPU可用")
    except Exception as e:
        st.error(f"❌ 连接失败: {str(e)}")

# 页脚
st.markdown("---")
st.markdown(f"""
*更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Project Hermes v2.0*
""")
