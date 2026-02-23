"""PDF export service for generating experiment reports."""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_experiment_report(experiment_data: dict, runs: list[dict]) -> bytes:
    """Generate a comprehensive PDF report for an experiment."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1E40AF'),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12,
    )

    story = []

    # Title
    story.append(Paragraph("Model Autopilot Experiment Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Experiment Info
    story.append(Paragraph("Experiment Overview", heading_style))
    exp_id = experiment_data.get('id', 'N/A')
    exp_id_short = exp_id[:8] + '...' if len(exp_id) > 8 else exp_id

    info_data = [
        ['Experiment ID', exp_id_short],
        ['Dataset ID', str(experiment_data.get('dataset_id', 'N/A'))[:8] + '...'],
        ['Status', experiment_data.get('status', 'N/A').upper()],
        ['Strategy', experiment_data.get('strategy', 'N/A')],
        ['Epsilon', f"{experiment_data.get('epsilon', 0.0):.2f}"],
        ['Created', experiment_data.get('created_at', 'N/A')],
        ['Finished', experiment_data.get('finished_at', 'N/A') or 'In Progress'],
        ['Total Runs', str(len(runs))],
    ]

    info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3 * inch))

    # Configuration
    config = experiment_data.get('config')
    if config:
        story.append(Paragraph("Configuration", heading_style))

        config_data = [
            ['Primary Metric', config.get('primary_metric', 'N/A')],
            ['Max Iterations', str(config.get('max_iterations', 'N/A'))],
            ['Test Size', f"{config.get('test_size', 0.2):.1%}"],
            ['Cross Validation', 'Yes' if config.get('cross_validation') else 'No'],
            ['Multi-Objective', 'Yes' if config.get('multi_objective') else 'No'],
        ]

        config_table = Table(config_data, colWidths=[2 * inch, 4 * inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
        ]))
        story.append(config_table)
        story.append(Spacer(1, 0.3 * inch))

    # Performance Chart
    if runs:
        story.append(Paragraph("Performance Trajectory", heading_style))
        chart_buffer = _create_performance_chart(runs, config.get('primary_metric', 'f1'))
        if chart_buffer:
            img = Image(chart_buffer, width=6 * inch, height=3.5 * inch)
            story.append(img)
            story.append(Spacer(1, 0.3 * inch))

    story.append(PageBreak())

    # Iteration Details
    story.append(Paragraph("Iteration Details", heading_style))

    for i, run in enumerate(runs, 1):
        story.append(Paragraph(f"<b>Iteration {run.get('iteration', i)}</b>", styles['Heading3']))

        iter_data = [
            ['Model', run.get('model_name', 'N/A')],
            ['Training Time', f"{run.get('train_time_s', 0):.2f}s"],
        ]

        # Add metrics
        metrics = run.get('metrics', {})
        for metric_name, metric_value in metrics.items():
            iter_data.append([metric_name.upper(), f"{metric_value:.4f}"])

        iter_table = Table(iter_data, colWidths=[2 * inch, 2 * inch])
        iter_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
        ]))
        story.append(iter_table)

        # Hyperparameters
        hparams = run.get('hyperparameters', {})
        if hparams:
            story.append(Spacer(1, 0.1 * inch))
            hp_text = "Hyperparameters: " + ", ".join([f"{k}={v}" for k, v in list(hparams.items())[:5]])
            story.append(Paragraph(f"<i>{hp_text}</i>", styles['Normal']))

        story.append(Spacer(1, 0.2 * inch))

    # Best Metrics Summary
    story.append(PageBreak())
    story.append(Paragraph("Best Metrics Achieved", heading_style))

    best_metrics = experiment_data.get('best_metrics', {})
    if best_metrics:
        best_data = [[metric.upper(), f"{value:.4f}"] for metric, value in best_metrics.items()]

        best_table = Table(best_data, colWidths=[3 * inch, 2 * inch])
        best_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
        ]))
        story.append(best_table)

    # Footer
    story.append(Spacer(1, 0.5 * inch))
    footer_text = f"Generated by Model Autopilot on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(f"<i>{footer_text}</i>", styles['Normal']))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _create_performance_chart(runs: list[dict], primary_metric: str) -> io.BytesIO | None:
    """Create a performance trajectory chart."""
    try:
        iterations = [r.get('iteration', i) for i, r in enumerate(runs, 1)]
        metric_values = [r.get('metrics', {}).get(primary_metric, 0) for r in runs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, metric_values, marker='o', linewidth=2, markersize=8, color='#3B82F6')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(primary_metric.upper(), fontsize=12)
        ax.set_title(f'{primary_metric.upper()} Performance Trajectory', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        return buffer
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None
