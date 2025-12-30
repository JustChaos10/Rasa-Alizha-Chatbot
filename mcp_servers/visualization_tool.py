"""
Visualization Tool - Schema-Driven Template Engine

Pure deterministic chart rendering using matplotlib templates.
NO LLM calls - uses predefined templates with professional styling.

Features:
- ChartRenderer class with hardcoded beautiful styles
- Unified entry point: create_chart_from_config(config)
- Supports: bar, pie, line charts
- Uses matplotlib with Agg backend (no GUI required)
"""

# CRITICAL: Set Agg backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import logging
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# PROFESSIONAL COLOR PALETTE
# ============================================================================

COLORS = {
    "primary": "#2563eb",      # Blue
    "success": "#16a34a",      # Green
    "warning": "#d97706",      # Orange/Amber
    "danger": "#dc2626",       # Red
    "info": "#0891b2",         # Cyan
    "purple": "#9333ea",       # Purple
    "pink": "#db2777",         # Pink
    "slate": "#475569",        # Slate gray
}

# Extended palette for charts with many categories
CHART_PALETTE = [
    "#2563eb", "#16a34a", "#d97706", "#dc2626", "#9333ea",
    "#0891b2", "#db2777", "#475569", "#84cc16", "#f59e0b",
    "#6366f1", "#14b8a6", "#f43f5e", "#8b5cf6", "#06b6d4"
]

# ============================================================================
# CHART RENDERER - Deterministic Templates
# ============================================================================

class ChartRenderer:
    """
    Renders charts using deterministic matplotlib templates.
    No LLM calls - pure template-based rendering with professional styling.
    """
    
    def __init__(self, charts_dir: str = "charts"):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set global matplotlib style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            # Fallback for older matplotlib
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass
        
        plt.rcParams.update({
            'font.family': ['Arial', 'sans-serif'],
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 16,
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2
        })
        
        logger.info(f"ChartRenderer initialized (charts_dir: {self.charts_dir})")
    
    def render_bar(
        self,
        data: Dict[str, float],
        title: str = "Comparison Chart",
        xlabel: str = "",
        ylabel: str = "Value",
        horizontal: bool = True,
        query_id: str = "default"
    ) -> Optional[str]:
        """
        Render a bar chart with professional styling.
        
        Args:
            data: Dictionary of {label: value}
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            horizontal: If True, render horizontal bars (better for long labels)
            query_id: Unique identifier for the file
            
        Returns:
            Path to saved chart image, or None on failure
        """
        if not data or len(data) < 1:
            logger.warning("Bar chart: insufficient data")
            return None
        
        try:
            labels = [str(k) for k in data.keys()]
            values = list(data.values())
            
            # Determine figure size based on number of items
            n_items = len(labels)
            if horizontal:
                fig_height = max(4, min(12, n_items * 0.5 + 2))
                fig, ax = plt.subplots(figsize=(10, fig_height))
            else:
                fig_width = max(8, min(16, n_items * 0.8 + 2))
                fig, ax = plt.subplots(figsize=(fig_width, 6))
            
            # Color gradient
            colors = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(n_items)]
            
            if horizontal:
                # Horizontal bar chart
                bars = ax.barh(range(n_items), values, color=colors, edgecolor='white', linewidth=0.5)
                ax.set_yticks(range(n_items))
                ax.set_yticklabels(labels)
                ax.invert_yaxis()  # Top to bottom
                ax.set_xlabel(ylabel)
                if xlabel:
                    ax.set_ylabel(xlabel)
                
                # Add value labels on bars
                max_val = max(values) if values else 1
                for bar, val in zip(bars, values):
                    width = bar.get_width()
                    label_text = f'{val:,.1f}' if val != int(val) else f'{int(val):,}'
                    ax.text(width + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                           label_text, va='center', ha='left', fontsize=9, color='#333')
            else:
                # Vertical bar chart
                bars = ax.bar(range(n_items), values, color=colors, edgecolor='white', linewidth=0.5)
                ax.set_xticks(range(n_items))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel(ylabel)
                if xlabel:
                    ax.set_xlabel(xlabel)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    label_text = f'{val:,.1f}' if val != int(val) else f'{int(val):,}'
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           label_text, ha='center', va='bottom', fontsize=9, color='#333')
            
            ax.set_title(title, pad=15)
            ax.grid(axis='x' if horizontal else 'y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            filename = f"bar_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.charts_dir / filename
            plt.savefig(save_path)
            plt.close(fig)
            
            logger.info(f"✅ Bar chart saved: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ Bar chart rendering failed: {e}")
            plt.close('all')
            return None
    
    def render_pie(
        self,
        data: Dict[str, float],
        title: str = "Distribution Chart",
        query_id: str = "default"
    ) -> Optional[str]:
        """
        Render a pie chart with professional styling.
        
        Args:
            data: Dictionary of {label: value}
            title: Chart title
            query_id: Unique identifier for the file
            
        Returns:
            Path to saved chart image, or None on failure
        """
        if not data or len(data) < 2:
            logger.warning("Pie chart: insufficient data (need at least 2 items)")
            return None
        
        try:
            labels = [str(k) for k in data.keys()]
            values = list(data.values())
            
            # Filter out zero/negative values
            filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
            if len(filtered) < 2:
                logger.warning("Pie chart: not enough positive values")
                return None
            
            labels, values = zip(*filtered)
            n_items = len(labels)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Colors
            colors = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(n_items)]
            
            # Slight explosion for emphasis
            explode = [0.02] * n_items
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                values,
                labels=None,  # We'll use legend instead
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=explode,
                pctdistance=0.75,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Style percentage labels
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            # Legend on the right
            ax.legend(
                wedges, labels,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=9
            )
            
            ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
            ax.axis('equal')
            
            plt.tight_layout()
            
            # Save
            filename = f"pie_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.charts_dir / filename
            plt.savefig(save_path)
            plt.close(fig)
            
            logger.info(f"✅ Pie chart saved: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ Pie chart rendering failed: {e}")
            plt.close('all')
            return None
    
    def render_line(
        self,
        data: Dict[str, float],
        title: str = "Trend Chart",
        xlabel: str = "Category",
        ylabel: str = "Value",
        query_id: str = "default"
    ) -> Optional[str]:
        """
        Render a line chart with professional styling.
        
        Args:
            data: Dictionary of {label: value} (ordered)
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            query_id: Unique identifier for the file
            
        Returns:
            Path to saved chart image, or None on failure
        """
        if not data or len(data) < 2:
            logger.warning("Line chart: insufficient data (need at least 2 points)")
            return None
        
        try:
            labels = [str(k) for k in data.keys()]
            values = list(data.values())
            n_points = len(labels)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot line with markers
            x = range(n_points)
            ax.plot(x, values, marker='o', linewidth=2.5, markersize=8,
                   color=COLORS["primary"], markerfacecolor='white',
                   markeredgecolor=COLORS["primary"], markeredgewidth=2)
            
            # Fill under the line
            ax.fill_between(x, values, alpha=0.15, color=COLORS["primary"])
            
            # Add value labels at each point
            for i, (xi, val) in enumerate(zip(x, values)):
                label_text = f'{val:,.1f}' if val != int(val) else f'{int(val):,}'
                ax.annotate(label_text, (xi, val), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, color='#333')
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add some padding to y-axis
            ymin, ymax = ax.get_ylim()
            padding = (ymax - ymin) * 0.15
            ax.set_ylim(ymin - padding * 0.5, ymax + padding)
            
            plt.tight_layout()
            
            # Save
            filename = f"line_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.charts_dir / filename
            plt.savefig(save_path)
            plt.close(fig)
            
            logger.info(f"✅ Line chart saved: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"❌ Line chart rendering failed: {e}")
            plt.close('all')
            return None


# ============================================================================
# UNIFIED ENTRY POINT
# ============================================================================

def create_chart_from_config(
    config: Dict[str, Any],
    charts_dir: str = "charts",
    query_id: str = "default"
) -> Optional[str]:
    """
    Unified entry point for chart generation from a config dict.
    
    This is the main function called by SQL/KB servers after LLM returns
    a visualization config in JSON format.
    
    Args:
        config: Dictionary with chart configuration:
            {
                "type": "bar" | "pie" | "line",
                "data": {"label1": value1, "label2": value2, ...},
                "title": "Optional chart title",
                "xlabel": "Optional x-axis label",
                "ylabel": "Optional y-axis label"
            }
        charts_dir: Directory to save charts
        query_id: Unique identifier for the file
        
    Returns:
        Path to saved chart image, or None on failure
    """
    if not config:
        logger.warning("create_chart_from_config: empty config")
        return None
    
    chart_type = config.get("type", "").lower()
    data = config.get("data", {})
    title = config.get("title", "Chart")
    xlabel = config.get("xlabel", "")
    ylabel = config.get("ylabel", "Value")
    
    if not chart_type or not data:
        logger.warning(f"create_chart_from_config: missing type or data. Config: {config}")
        return None
    
    # Ensure data is a dict with numeric values
    if isinstance(data, list):
        # Convert list of dicts to single dict
        try:
            if data and isinstance(data[0], dict):
                # Assume format [{"name": "A", "value": 1}, ...]
                data = {item.get("name", item.get("label", f"Item {i}")): 
                       float(item.get("value", item.get("count", 0)))
                       for i, item in enumerate(data)}
        except Exception as e:
            logger.warning(f"Failed to convert data list to dict: {e}")
            return None
    
    # Validate data values are numeric
    try:
        data = {str(k): float(v) for k, v in data.items()}
    except (ValueError, TypeError) as e:
        logger.warning(f"Data values must be numeric: {e}")
        return None
    
    # Create renderer and generate chart
    renderer = ChartRenderer(charts_dir=charts_dir)
    
    if chart_type == "bar":
        return renderer.render_bar(data, title=title, xlabel=xlabel, ylabel=ylabel, query_id=query_id)
    elif chart_type == "pie":
        return renderer.render_pie(data, title=title, query_id=query_id)
    elif chart_type == "line":
        return renderer.render_line(data, title=title, xlabel=xlabel, ylabel=ylabel, query_id=query_id)
    else:
        logger.warning(f"Unknown chart type: {chart_type}")
        return None


# ============================================================================
# DATA PARSING UTILITIES
# ============================================================================

def parse_sql_result_to_data(result: str) -> Dict[str, float]:
    """
    Parse SQL query result text into a dictionary suitable for charting.
    
    Handles common formats:
    - Tuple format: ('Label', Decimal('123.45'))
    - Pipe-separated: Label | 123.45
    - Colon-separated: Label: 123.45
    
    Returns:
        Dictionary of {label: value}
    """
    data = {}
    
    if not result or not result.strip():
        return data
    
    # Strategy 1: SQL Tuple Format - ('Label', Decimal('123.45')) or ('Label', 123)
    tuple_matches = re.findall(r"\('([^']+)'[,\s]+(?:Decimal\(')?(\d+(?:\.\d+)?)", result)
    if tuple_matches and len(tuple_matches) >= 2:
        for label, val in tuple_matches[:15]:  # Limit to 15 items
            try:
                data[label.strip()] = float(val.replace(',', ''))
            except ValueError:
                continue
        if data:
            logger.debug(f"Parsed {len(data)} items using tuple format")
            return data
    
    # Strategy 2: Simple tuple - ('Name', 123)
    simple_tuples = re.findall(r"\('([^']+)',\s*(\d+(?:\.\d+)?)\)", result)
    if simple_tuples and len(simple_tuples) >= 2:
        for label, val in simple_tuples[:15]:
            try:
                data[label.strip()] = float(val.replace(',', ''))
            except ValueError:
                continue
        if data:
            logger.debug(f"Parsed {len(data)} items using simple tuple format")
            return data
    
    # Strategy 3: Pipe-separated - Label | 123
    lines = result.strip().split('\n')
    for line in lines:
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                label = parts[0].strip()
                # Find numeric value in remaining parts
                for part in parts[1:]:
                    match = re.search(r'(\d+(?:[,.]?\d+)*)', part.strip())
                    if match:
                        try:
                            val = float(match.group(1).replace(',', ''))
                            if label and not label.startswith('-'):
                                data[label] = val
                                break
                        except ValueError:
                            continue
    if data and len(data) >= 2:
        logger.debug(f"Parsed {len(data)} items using pipe format")
        return dict(list(data.items())[:15])
    
    # Strategy 4: Colon-separated - Label: 123
    data = {}
    for line in lines:
        if ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                label = parts[0].strip()
                match = re.search(r'(\d+(?:[,.]?\d+)*)', parts[-1])
                if match and label:
                    try:
                        val = float(match.group(1).replace(',', ''))
                        data[label] = val
                    except ValueError:
                        continue
    if data and len(data) >= 2:
        logger.debug(f"Parsed {len(data)} items using colon format")
        return dict(list(data.items())[:15])
    
    # Strategy 5: Word followed by number pattern
    data = {}
    all_matches = re.findall(r"([A-Za-z][A-Za-z\s]{1,30})\D*?(\d+(?:[,.]?\d+)*)", result)
    for label, val in all_matches:
        label = label.strip()
        if label and len(label) > 1:
            try:
                data[label] = float(val.replace(',', ''))
            except ValueError:
                continue
    if data and len(data) >= 2:
        logger.debug(f"Parsed {len(data)} items using word-number pattern")
        return dict(list(data.items())[:15])
    
    logger.debug("No parseable data found in result")
    return {}


def detect_chart_type(query: str, data: Dict[str, float]) -> Optional[str]:
    """
    Detect appropriate chart type based on query keywords and data shape.
    Prefers line charts for time-series/sequential data.
    
    Args:
        query: User's original query
        data: Parsed data dictionary
        
    Returns:
        Chart type string ("bar", "pie", "line") or None
    """
    if not data or len(data) < 2:
        return None
    
    query_lower = query.lower()
    labels = list(data.keys())
    
    # Explicit chart requests (highest priority)
    if "pie chart" in query_lower or "pie graph" in query_lower:
        return "pie"
    if "bar chart" in query_lower or "bar graph" in query_lower:
        return "bar"
    if "line chart" in query_lower or "line graph" in query_lower:
        return "line"
    
    # Check if data looks like time-series (prefer line chart)
    time_patterns = [
        r'\d{4}',  # Years like 2020, 2021
        r'\d{4}-\d{2}',  # Year-month like 2020-01
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',  # Month names
        r'(q1|q2|q3|q4)',  # Quarters
        r'(week|month|year|day)',  # Time period words
        r'\d{1,2}/\d{1,2}',  # Date patterns
    ]
    import re
    is_time_series = False
    for label in labels:
        label_lower = str(label).lower()
        for pattern in time_patterns:
            if re.search(pattern, label_lower):
                is_time_series = True
                break
        if is_time_series:
            break
    
    # If time-series data detected, prefer line chart
    if is_time_series:
        return "line"
    
    # Infer from query keywords
    line_keywords = ["trend", "over time", "growth", "change", "timeline", "history", 
                     "monthly", "yearly", "quarterly", "daily", "weekly", "by month",
                     "by year", "per month", "per year", "time", "period", "progression"]
    if any(kw in query_lower for kw in line_keywords):
        return "line"
    
    pie_keywords = ["distribution", "breakdown", "share", "percentage", "portion", "composition"]
    if any(kw in query_lower for kw in pie_keywords):
        return "pie"
    
    # Default to bar for comparisons
    bar_keywords = ["compare", "top", "highest", "lowest", "best", "most", "least", "count", "number"]
    if any(kw in query_lower for kw in bar_keywords):
        return "bar"
    
    # If generic chart request and data has >5 items, consider line for readability
    if any(word in query_lower for word in ["chart", "graph", "visualize", "plot"]):
        if len(data) > 5:
            return "line"  # Line charts handle many data points better
        return "bar"
    
    return None


# ============================================================================
# SMART VISUALIZATION TOOL - LLM-Powered Chart Decision
# ============================================================================

class SmartVisualizationTool:
    """
    LLM-powered visualization tool that intelligently decides when and how to visualize data.
    
    Unlike the template-based VisualizationTool, this class:
    1. Uses LLM to analyze query intent and data structure
    2. Automatically decides optimal chart type
    3. Extracts chartable data from text (including KB answers)
    4. Generates visualizations without explicit user keywords
    
    Usage:
        viz_tool = SmartVisualizationTool(llm, charts_dir="charts/kb")
        result = await viz_tool.create_visualization(query, answer, state_id)
    """
    
    def __init__(self, llm, charts_dir: str = "charts"):
        """
        Initialize SmartVisualizationTool.
        
        Args:
            llm: MultiProviderLLM instance for making decisions
            charts_dir: Directory to save charts
        """
        self.llm = llm
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = ChartRenderer(charts_dir=str(self.charts_dir))
        logger.info(f"SmartVisualizationTool initialized (charts_dir: {self.charts_dir})")
    
    async def create_visualization(
        self,
        query: str,
        result: str,
        query_id: str = "default",
        force_visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze query and result to create appropriate visualization.
        
        Args:
            query: User's original query
            result: The answer/data to potentially visualize
            query_id: Unique identifier for the chart file
            force_visualize: If True, try harder to create visualization
            
        Returns:
            Dict with keys:
                - success: bool
                - chart_path: str or None
                - chart_type: str or None
                - message: str (explanation)
        """
        try:
            # Step 1: Try to parse data from result
            data = parse_sql_result_to_data(result)
            
            # Step 1.5: Translate data labels if query is Arabic and data exists
            is_arabic = any('\u0600' <= c <= '\u06FF' for c in query)
            if is_arabic and data:
                data = await self._llm_translate_labels(data)
            
            # If no data parsed and not forced, skip visualization
            if not data and not force_visualize:
                return {
                    "success": False,
                    "chart_path": None,
                    "chart_type": None,
                    "message": "No chartable data found in result"
                }
            
            # Step 2: If force_visualize but no data, use LLM to extract data
            if not data and force_visualize:
                data = await self._llm_extract_data(query, result)
            
            if not data or len(data) < 2:
                return {
                    "success": False,
                    "chart_path": None,
                    "chart_type": None,
                    "message": "Insufficient data points for visualization (need at least 2)"
                }
            
            # Step 3: Determine chart type - ALWAYS use LLM for best decision
            chart_type = await self._llm_decide_chart_type(query, data)
            
            # Step 4: Generate the chart
            title = self._generate_title(query, chart_type)
            
            config = {
                "type": chart_type,
                "data": data,
                "title": title,
                "ylabel": "Value"
            }
            
            chart_path = create_chart_from_config(
                config,
                charts_dir=str(self.charts_dir),
                query_id=query_id
            )
            
            if chart_path:
                logger.info(f"SmartVisualizationTool: Created {chart_type} chart at {chart_path}")
                return {
                    "success": True,
                    "chart_path": chart_path,
                    "chart_type": chart_type,
                    "message": f"Created {chart_type} chart with {len(data)} data points"
                }
            else:
                return {
                    "success": False,
                    "chart_path": None,
                    "chart_type": chart_type,
                    "message": "Chart rendering failed"
                }
                
        except Exception as e:
            logger.error(f"SmartVisualizationTool error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "chart_path": None,
                "chart_type": None,
                "message": f"Error: {str(e)}"
            }
    
    async def _llm_extract_data(self, query: str, result: str) -> Dict[str, float]:
        """
        Use LLM to extract chartable data from text when parsing fails.
        """
        prompt = f"""Extract numeric data from this text for visualization.

TEXT:
{result[:2000]}

USER QUERY: {query}

Return ONLY a JSON object with labels and numeric values suitable for charting.
Example: {{"Category A": 10, "Category B": 25, "Category C": 15}}

If the text contains percentages, rates, or tax brackets, extract them.
For tax slabs like "0-4 lakh: Nil" treat "Nil" as 0.
For "5 per cent" extract as 5.

JSON:"""
        
        try:
            content = await self.llm.call_async(prompt, trace_name="viz-extract-data")
            # Try to parse JSON from response
            content = content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            data = json.loads(content)
            if isinstance(data, dict):
                # Validate all values are numeric
                clean_data = {}
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        clean_data[str(k)] = float(v)
                    elif v is None or str(v).lower() in ['nil', 'none', 'null']:
                        clean_data[str(k)] = 0.0
                return clean_data
        except Exception as e:
            logger.warning(f"LLM data extraction failed: {e}")
        
        return {}
    
    async def _llm_decide_chart_type(self, query: str, data: Dict[str, float]) -> str:
        """
        Use LLM to decide the best chart type for the data.
        """
        prompt = f"""Choose the best chart type for this data.

DATA: {json.dumps(data)}
USER QUERY: {query}

Options:
- "bar": For comparing categories, rankings, counts
- "pie": For showing parts of a whole, percentages, distributions
- "line": For trends over time, sequential data

Respond with ONLY ONE WORD: bar, pie, or line"""
        
        try:
            content = await self.llm.call_async(prompt, trace_name="viz-chart-type")
            chart_type = content.strip().lower()
            if chart_type in ["bar", "pie", "line"]:
                return chart_type
        except Exception as e:
            logger.warning(f"LLM chart type decision failed: {e}")
        
        return "bar"  # Default
    
    async def _llm_translate_labels(self, data: Dict[str, float]) -> Dict[str, float]:
        """
        Translate Arabic labels to English for better chart rendering.
        
        Args:
            data: Dictionary with potentially Arabic labels as keys
            
        Returns:
            Dictionary with translated English labels
        """
        if not data:
            return data
        
        labels = list(data.keys())
        prompt = f"""Translate these Arabic labels to concise English labels for a chart.

Labels: {labels}

Return ONLY a JSON object mapping Arabic to English.
Example: {{"العنصر أ": "Item A", "العنصر ب": "Item B"}}

JSON:"""
        
        try:
            content = await self.llm.call_async(prompt, trace_name="viz-translate")
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            translations = json.loads(content)
            if isinstance(translations, dict):
                translated_data = {}
                for k, v in data.items():
                    new_key = translations.get(k, k)
                    translated_data[new_key] = v
                return translated_data
        except Exception as e:
            logger.warning(f"LLM label translation failed: {e}")
        
        return data  # Return original if translation fails
    
    def _generate_title(self, query: str, chart_type: str) -> str:
        """Generate a reasonable chart title from the query (supports Arabic and English)."""
        # Clean up query for title
        query_clean = query.strip().rstrip('?').rstrip('؟').strip()
        
        # Shorten if too long
        if len(query_clean) > 50:
            words = query_clean.split()
            if len(words) > 5:
                query_clean = ' '.join(words[:5]) + "..."
            else:
                query_clean = query_clean[:47] + "..."
        
        # Capitalize first letter (only for English/Latin text)
        if query_clean and ord(query_clean[0]) < 128:
            query_clean = query_clean[0].upper() + query_clean[1:]
        
        return query_clean or f"{chart_type.capitalize()} Chart"


# ============================================================================
# LEGACY COMPATIBILITY - VisualizationTool wrapper
# ============================================================================

class VisualizationTool:
    """
    Legacy wrapper for backward compatibility with existing server code.
    
    Delegates to the new ChartRenderer and create_chart_from_config functions.
    """
    
    def __init__(self, charts_dir: str = "charts/kb", llm_model: str = None):
        """
        Initialize visualization tool.
        
        Args:
            charts_dir: Directory to save charts
            llm_model: Ignored (no LLM calls in new implementation)
        """
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = ChartRenderer(charts_dir=str(self.charts_dir))
        logger.info(f"VisualizationTool initialized (template-based, no LLM)")
    
    async def should_visualize(self, query: str, result: str, timeout: int = 60) -> str:
        """
        Determine if visualization is needed (heuristic-based).
        
        DEPRECATED: Use detect_chart_type() instead.
        
        Returns:
            "YES_BAR", "YES_PIE", "YES_LINE", or "NO"
        """
        data = parse_sql_result_to_data(result)
        chart_type = detect_chart_type(query, data)
        
        if chart_type:
            return f"YES_{chart_type.upper()}"
        return "NO"
    
    async def generate_visualization(
        self,
        query: str,
        result: str,
        visualization_type: str,
        query_id: str = None,
        timeout: int = 120
    ) -> Optional[str]:
        """
        Generate visualization from query result.
        
        DEPRECATED: Use create_chart_from_config() instead.
        """
        if visualization_type == "NO":
            return None
        
        chart_type = visualization_type.replace("YES_", "").lower()
        data = parse_sql_result_to_data(result)
        
        if not data:
            logger.warning("No data parsed for visualization")
            return None
        
        config = {
            "type": chart_type,
            "data": data,
            "title": "Query Results"
        }
        
        return create_chart_from_config(
            config,
            charts_dir=str(self.charts_dir),
            query_id=query_id or "default"
        )
    
    async def process_result(
        self,
        query: str,
        result: str,
        query_id: str = None
    ) -> Tuple[str, Optional[str]]:
        """
        Process query result and optionally generate visualization.
        
        DEPRECATED: Servers should use create_chart_from_config() directly.
        
        Returns:
            Tuple of (result_text, chart_path)
        """
        viz_type = await self.should_visualize(query, result)
        chart_path = None
        
        if viz_type != "NO":
            chart_path = await self.generate_visualization(
                query, result, viz_type, query_id
            )
        
        return result, chart_path
