"""
Utility functions for Agentic BI System
Display helpers for Jupyter notebook output
"""
import json
from html import escape
from typing import Any
from IPython.display import display, HTML
import pandas as pd


def log_agent_title_html(title: str, icon: str = "🤖") -> None:
    """Display agent title with icon"""
    display(HTML(f"""
      <div style="padding:1em;margin:1em 0;background-color:#f0f4f8;border-left:6px solid #1976D2;">
        <h2 style="margin:0;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
          {escape(icon)} {escape(title)}
        </h2>
      </div>
    """))


def log_tool_call_html(tool_name: str, arguments: Any) -> None:
    """Display tool call information"""
    display(HTML(f"""
      <div style="border-left:4px solid #1976D2;padding:.8em;margin:1em 0;
                  background-color:#e3f2fd;color:#0D47A1;font-family:'Segoe UI',sans-serif;">
        <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">
          🔧 <span style="color:#0B3D91;">Tool Call:</span> <span style="color:#0B3D91;">{escape(str(tool_name))}</span>
        </div>
        <code style="display:block;background:#e8f0fe;color:#1b1b1b;padding:6px;border-radius:4px;
                     font-size:13px;white-space:pre-wrap;">{escape(str(arguments))}</code>
      </div>
    """))


def log_tool_result_html(result: Any) -> None:
    """Display tool result"""
    result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
    # Truncate if too long
    if len(result_str) > 1000:
        result_str = result_str[:1000] + "\n... [truncated]"
    
    display(HTML(f"""
      <div style="border-left:4px solid #558B2F;padding:.8em;margin:1em 0;
                  background-color:#f1f8e9;color:#33691E;">
        <strong>✅ Tool Result:</strong>
        <pre style="white-space:pre-wrap;font-size:13px;color:#2E7D32;max-height:300px;overflow:auto;">{escape(result_str)}</pre>
      </div>
    """))


def log_final_summary_html(content: str) -> None:
    """Display final agent summary"""
    display(HTML(f"""
      <div style="border-left:4px solid #2E7D32;padding:1em;margin:1em 0;
                  background-color:#e8f5e9;color:#1B5E20;">
        <strong>✅ Agent Output:</strong>
        <pre style="white-space:pre-wrap;font-size:13px;color:#1B5E20;">{escape(content.strip())}</pre>
      </div>
    """))


def log_handoff_html(from_agent: str, to_agent: str, data_summary: str = "") -> None:
    """Display agent handoff information"""
    display(HTML(f"""
      <div style="border-left:4px solid #FF9800;padding:1em;margin:1em 0;
                  background-color:#fff3e0;color:#E65100;">
        <strong>🔄 Handoff:</strong> {escape(from_agent)} → {escape(to_agent)}
        <pre style="white-space:pre-wrap;font-size:12px;color:#BF360C;">{escape(data_summary[:500]) if data_summary else "Data passed to next agent"}</pre>
      </div>
    """))


def log_validation_html(status: str, message: str) -> None:
    """Display validation result"""
    if status == "passed":
        bg_color = "#e8f5e9"
        border_color = "#4CAF50"
        icon = "✅"
    else:
        bg_color = "#ffebee"
        border_color = "#f44336"
        icon = "❌"
    
    display(HTML(f"""
      <div style="border-left:4px solid {border_color};padding:1em;margin:1em 0;
                  background-color:{bg_color};">
        <strong>{icon} Validation {status.upper()}:</strong>
        <p style="margin:5px 0;font-size:14px;">{escape(message)}</p>
      </div>
    """))


def log_unexpected_html() -> None:
    """Display unexpected response warning"""
    display(HTML("""
      <div style="border-left:4px solid #F57C00;padding:1em;margin:1em 0;
                  background-color:#fff3e0;color:#E65100;">
        <strong>⚠️ Unexpected:</strong> No tool_calls or content returned.
      </div>
    """))


def display_insights_html(insights: list) -> None:
    """Display business insights as cards"""
    cards_html = ""
    for i, insight in enumerate(insights, 1):
        cards_html += f"""
        <div style="border:1px solid #e0e0e0;border-radius:8px;padding:15px;margin:10px 0;
                    background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:white;">
          <h4 style="margin:0 0 10px 0;">💡 Insight #{i}</h4>
          <p style="margin:0;font-size:14px;">{escape(str(insight))}</p>
        </div>
        """
    
    display(HTML(f"""
      <div style="font-family:'Segoe UI',sans-serif;">
        <h3 style="color:#333;">📊 Business Insights</h3>
        {cards_html}
      </div>
    """))


def display_recommendations_html(recommendations: list) -> None:
    """Display recommendations as action items"""
    items_html = ""
    for i, rec in enumerate(recommendations, 1):
        items_html += f"""
        <div style="display:flex;align-items:flex-start;margin:10px 0;padding:10px;
                    background-color:#f5f5f5;border-radius:6px;">
          <span style="background:#1976D2;color:white;border-radius:50%;width:24px;height:24px;
                       display:flex;align-items:center;justify-content:center;margin-right:10px;
                       font-size:12px;flex-shrink:0;">{i}</span>
          <p style="margin:0;font-size:14px;color:#333;">{escape(str(rec))}</p>
        </div>
        """
    
    display(HTML(f"""
      <div style="font-family:'Segoe UI',sans-serif;">
        <h3 style="color:#333;">🎯 Recommendations</h3>
        {items_html}
      </div>
    """))


def display_architecture_html() -> None:
    """Display system architecture diagram using ASCII/HTML"""
    display(HTML("""
    <div style="font-family:monospace;background:#1a1a2e;color:#eee;padding:20px;border-radius:10px;margin:10px 0;">
      <h3 style="color:#00d9ff;text-align:center;">🏗️ Agentic BI System Architecture</h3>
      <pre style="font-size:12px;line-height:1.4;">
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC BI SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │   📊 DATA    │      │   🔬 DATA    │      │   💼 BUSINESS│              │
│  │   ENGINEER   │ ───▶ │   SCIENTIST  │ ───▶ │   MANAGER   │              │
│  │    AGENT     │      │    AGENT     │      │    AGENT    │              │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘              │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │  TOOLS:      │      │  TOOLS:      │      │  TOOLS:      │              │
│  │ • read_csv   │      │ • analysis   │      │ • KPIs       │              │
│  │ • summarize  │      │ • KPIs       │      │ • report     │              │
│  │              │      │ • anomalies  │      │              │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                              │
│         │                     │                     │                       │
│         └──────────┬──────────┴──────────┬──────────┘                       │
│                    ▼                     ▼                                  │
│            ┌──────────────┐      ┌──────────────┐                          │
│            │  ✅ VALIDATOR │      │  📋 FINAL    │                          │
│            │    AGENT     │ ───▶ │   OUTPUT    │                          │
│            └──────────────┘      └──────────────┘                          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  DATA SOURCE: stripe_payments_dataset.csv                                   │
│  LLM: OpenAI GPT-4 Mini                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
      </pre>
    </div>
    """))


def display_final_report_html(report: dict) -> None:
    """Display final executive report"""
    display(HTML(f"""
    <div style="font-family:'Segoe UI',sans-serif;border:2px solid #1976D2;border-radius:12px;padding:20px;margin:10px 0;">
      <h2 style="color:#1976D2;text-align:center;border-bottom:2px solid #e0e0e0;padding-bottom:10px;">
        📈 Executive Business Report
      </h2>
      
      <div style="background:#f5f5f5;padding:15px;border-radius:8px;margin:15px 0;">
        <h3 style="margin:0 0 10px 0;color:#333;">📊 Summary</h3>
        <pre style="white-space:pre-wrap;font-size:13px;">{escape(json.dumps(report.get('summary', {}), indent=2))}</pre>
      </div>
      
      <div style="background:#e3f2fd;padding:15px;border-radius:8px;margin:15px 0;">
        <h3 style="margin:0 0 10px 0;color:#1976D2;">💡 Key Insights</h3>
        <pre style="white-space:pre-wrap;font-size:13px;">{escape(json.dumps(report.get('insights', []), indent=2))}</pre>
      </div>
      
      <div style="background:#e8f5e9;padding:15px;border-radius:8px;margin:15px 0;">
        <h3 style="margin:0 0 10px 0;color:#2E7D32;">🎯 Recommendations</h3>
        <pre style="white-space:pre-wrap;font-size:13px;">{escape(json.dumps(report.get('recommendations', []), indent=2))}</pre>
      </div>
    </div>
    """))
