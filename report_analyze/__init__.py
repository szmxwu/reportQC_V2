"""
Report analyze package for report quality checking.
"""

from .report_conclusion_checker import check_report_conclusion
from .contradiction_checker import check_contradiction

__all__ = ['check_report_conclusion', 'check_contradiction']
