#!/usr/bin/env python3
"""
Core Pipeline Components v5.0.0
===============================

This module contains the core unified components for pipeline execution,
including the PipelineExecutor that eliminates code duplication between
run_pipeline.py and src/main.py.

Components:
- PipelineExecutor: Unified pipeline execution and management
"""

from .pipeline_executor import PipelineExecutor, get_pipeline_executor

__version__ = "5.0.0"
__all__ = ["PipelineExecutor", "get_pipeline_executor"]