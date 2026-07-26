"""
vcf_agent package

- Provides tools and wrappers for VCF/BCF analysis, validation, and compliance
- Integrates bcftools, GATK, and AI models
- Exposes CLI and API interfaces for genomics workflows
"""

__all__ = [
    "agent",
    "SYSTEM_PROMPT",
    "get_agent_with_session",
    "run_llm_analysis_task"
]

__version__ = "0.1.0" # Or your current project version

# You can also expose key components of your library here if desired
# For example:
# from .agent import VCFAnalysisAgent, get_agent_with_session
# from .cli import main as run_cli

print(f"VCF Analysis Agent initialized, version {__version__}") 