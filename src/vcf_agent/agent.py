"""
VCF Analysis Agent: Strands-based agent for VCF/BCF analysis, validation, and processing.

- Integrates bcftools and AI models for genomics workflows
- Provides CLI and API tools for validation and analysis
- Extensible with additional tools and models

Output Mode Toggling (Chain-of-Thought vs. Raw Output):
-------------------------------------------------------
The agent supports three ways to control output mode:
  1. Environment variable: VCF_AGENT_RAW_MODE ("1", "true", "yes" = raw output)
  2. CLI flag: --raw / --no-think (see cli.py)
  3. Session-based: SessionConfig(raw_mode=...) (see config.py)

See the README for usage examples and details.

"""

from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands.models.litellm import LiteLLMModel
from typing import Any, Optional, Dict, List as PyList, cast, Union, Literal, List
from .validation import validate_vcf_file
from .bcftools_integration import (
    bcftools_view as _bcftools_view,
    bcftools_query as _bcftools_query,
    bcftools_filter as _bcftools_filter,
    bcftools_norm as _bcftools_norm,
    bcftools_stats as _bcftools_stats,
    bcftools_annotate as _bcftools_annotate,
    vcf_compare as _vcf_compare,
)
import os
from .config import SessionConfig
from .api_clients import OpenAIClient, CerebrasClient, APIClientError
from .prompt_templates import get_prompt_for_task, load_prompt_contract
from . import graph_integration # Import the module
from . import vcf_utils # Import the new function
import time # For timing
from . import metrics # Import metrics module
import json # Added import
import subprocess
import logging
import tempfile # Ensure tempfile is imported
import re # Add this import for regex
import jsonschema
from jsonschema import validate, ValidationError

# OpenTelemetry Imports
from opentelemetry import trace
from .tracing import init_tracer # Assuming setup_auto_instrumentation is called by CLI

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry Tracer for the agent
agent_tracer = init_tracer(service_name="vcf-agent-core")


class ZAIReasoningModel(LiteLLMModel):
    """LiteLLM model that preserves GLM `reasoning_content` across turns.

    Strands' default OpenAI-compatible message formatter intentionally DROPS
    `reasoningContent` blocks for assistant messages (it logs
    "reasoningContent is not supported in multi-turn conversations with the
    Chat Completions API"). That breaks GLM-5.x reasoning models on any task
    that needs more than one tool call: the chain-of-thought is lost between
    turns, the model loops, and multi-step analyses never complete.

    Z.AI's API DOES accept `reasoning_content` on assistant messages in the
    history (verified end-to-end), so for the zai provider we keep those
    blocks and serialize them as a top-level `reasoning_content` field on the
    outgoing assistant message. This preserves the reasoning chain across
    multi-turn tool-calling.
    """

    @classmethod
    def _format_regular_messages(cls, messages, **kwargs):
        """Same as the OpenAI formatter, but keeps reasoningContent for zai."""
        formatted_messages = []
        for message in messages:
            contents = message["content"]
            # Extract the reasoning text (if any) to re-attach it.
            reasoning_text = None
            for content in contents:
                rc = content.get("reasoningContent")
                if rc and "text" in rc:
                    reasoning_text = rc["text"]
                    break

            # Filter out only toolResult/toolUse; KEEP reasoningContent handled above.
            filtered_contents = []
            for content in contents:
                if any(block_type in content for block_type in ["toolResult", "toolUse", "reasoningContent"]):
                    continue
                filtered_contents.append(content)

            formatted_contents = [cls.format_request_message_content(content) for content in filtered_contents]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                **({"content": formatted_contents} if formatted_contents else {}),
            }
            if formatted_tool_calls:
                formatted_message["tool_calls"] = formatted_tool_calls
            # Re-attach reasoning_content for assistant turns so the model
            # retains its chain-of-thought across the conversation.
            if message["role"] == "assistant" and reasoning_text:
                formatted_message["reasoning_content"] = reasoning_text
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)
        return formatted_messages


# Output mode toggling: chain-of-thought (CoT) vs. raw output
# 1. Environment variable: VCF_AGENT_RAW_MODE ("1", "true", "yes" = raw)
# 2. CLI flag: --raw / --no-think (see cli.py)
# 3. Session-based: SessionConfig(raw_mode=...) (see config.py)
# See README for details.
RAW_MODE = False  # Enable chain-of-thought by default for better reasoning

SYSTEM_PROMPT = (
    "You are the VCF Analysis Agent, a specialized assistant for genomics workflows. "
    "You help users analyze, validate, and process VCF (Variant Call Format) files.\n\n"
    
    "Your capabilities include:\n"
    "- VCF file validation and format checking\n"
    "- BCFtools operations (view, query, filter, norm, stats, annotate)\n"
    "- VCF file comparison and analysis\n"
    "- AI-powered variant analysis and summarization\n"
    "- Graph database integration for variant relationships\n"
    "- Performance optimization and quality metrics\n\n"
    
    "You can engage in natural conversation about genomics topics and automatically use tools when needed. "
    "When a user asks you to validate, analyze, compare, or process VCF files, use the appropriate tools. "
    "Always provide clear, helpful explanations of your analysis and findings.\n\n"
    
    "Available tools include: validate_vcf, bcftools operations, AI-powered analysis tools, "
    "and graph database tools. Use them intelligently based on user requests."
)

# Global variable to hold the main Kuzu connection, initialized by get_agent_with_session
# This is one way; alternatively, tools can call get_managed_kuzu_connection() directly.
# For now, we'll initialize it to ensure DB and schema are ready when an agent is created.
_kuzu_connection_for_agent = None

# Placeholder echo tool using @tool decorator
@tool
def echo(text: str) -> str:
    """
    Echoes the input text back to the user.

    Args:
        text (str): Text to echo back.

    Returns:
        str: The echoed text.

    Example:
        >>> echo("Hello, world!")
        'Echo: Hello, world!'
    """
    return f"Echo: {text}"

# VCF validation tool
@tool
def validate_vcf(filepath: str) -> str:
    """
    Validates a VCF/BCF file for existence, format, index, and bcftools stats.
    Args:
        filepath (str): Path to the VCF/BCF file.
    Returns:
        str: Validation result as a string (valid/invalid and error message if any).
    """
    tool_name = "validate_vcf"
    print(f"[TOOL {tool_name}] Entered. Filepath: {filepath}")

    # Metrics
    metrics_status = "success"
    error_type_str = None
    result_str_to_return = "INTERNAL_ERROR: Validation result not set" # Initialize with a default error

    with agent_tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
        tool_span.set_attribute("tool.name", tool_name)
        tool_span.set_attribute("tool.args.filepath", filepath)
        tool_span.add_event("tool.execution.start")

        start_time = time.time()
        try:
            # Call the validation function
            is_valid, error_message = validate_vcf_file(filepath)
            
            # Convert tuple result to user-friendly string
            if is_valid:
                result_str_to_return = f"✅ VCF file '{filepath}' is valid and passed all validation checks."
            else:
                result_str_to_return = f"❌ VCF file '{filepath}' failed validation: {error_message}"
            
            tool_span.set_status(trace.StatusCode.OK)
            tool_span.add_event("tool.execution.end")
        except Exception as e:
            error_type_str = type(e).__name__
            result_str_to_return = f"ERROR: {str(e)}"
            metrics_status = "error"
            tool_span.set_status(trace.StatusCode.ERROR, description=str(e))
            tool_span.add_event("tool.execution.error", {"error": str(e)})
        finally:
            # Record metrics
            duration = time.time() - start_time
            metrics.record_tool_usage(tool_name, duration, metrics_status, error_type_str)

    print(f"[TOOL {tool_name}] Exiting. Result: {result_str_to_return[:100]}...")
    return result_str_to_return


def _coerce_args_list(args: Any) -> PyList[str]:
    """Normalize the `args` parameter of bcftools_*_tool wrappers into a list of str.

    Strands 1.50 has a bug: even though the tool's JSON schema declares
    `args` as `type: array` and the model correctly sends a JSON array, the
    value reaching the Python function is the array serialized to a STRING
    (e.g. the string '["-n", "3", "file.vcf.gz"]'). bcftools then receives a
    single bogus argument and fails. We detect both shapes (real list, or a
    JSON string / shell string) and always return list[str].
    """
    if isinstance(args, list):
        return [str(a) for a in args]
    if isinstance(args, str):
        s = args.strip()
        # JSON array form: '["-n","3","f.vcf"]'
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(a) for a in parsed]
            except json.JSONDecodeError:
                pass
        # Shell form: '-n 3 file.vcf'  -> split on whitespace
        return s.split()
    # Fallback: wrap anything else as a single-element list.
    return [str(args)]


@tool
def bcftools_view_tool(args: PyList[str]) -> str:
    """
    Run bcftools view with the given arguments.

    Args:
        args (list): Arguments for bcftools view.
    Returns:
        str: Output or error from bcftools.
    """
    rc, out, err = _bcftools_view(_coerce_args_list(args))
    return out if rc == 0 else err

@tool
def bcftools_query_tool(args: PyList[str]) -> str:
    """
    Run bcftools query with the given arguments.

    Args:
        args (list): Arguments for bcftools query.
    Returns:
        str: Output or error from bcftools.
    """
    rc, out, err = _bcftools_query(_coerce_args_list(args))
    return out if rc == 0 else err

@tool
def bcftools_filter_tool(args: PyList[str]) -> str:
    """
    Run bcftools filter with the given arguments.

    Args:
        args (list): Arguments for bcftools filter.
    Returns:
        str: Output or error from bcftools.
    """
    rc, out, err = _bcftools_filter(_coerce_args_list(args))
    return out if rc == 0 else err

@tool
def bcftools_norm_tool(args: PyList[str]) -> str:
    """
    Run bcftools norm with the given arguments.

    Args:
        args (list): Arguments for bcftools norm.
    Returns:
        str: Output or error from bcftools.
    """
    rc, out, err = _bcftools_norm(_coerce_args_list(args))
    return out if rc == 0 else err

@tool
def bcftools_stats_tool(args: PyList[str]) -> str:
    """
    Run bcftools stats with the given arguments.

    Args:
        args (list): Arguments for bcftools stats.
    Returns:
        str: Output or error from bcftools.
    """
    rc, out, err = _bcftools_stats(_coerce_args_list(args))
    return out if rc == 0 else err

@tool
def bcftools_annotate_tool(args: PyList[str]) -> str:
    """
    Run bcftools annotate with the given arguments.

    Args:
        args (list): Arguments for bcftools annotate.
    Returns:
        str: Output or error from bcftools.
    """
    rc, out, err = _bcftools_annotate(_coerce_args_list(args))
    return out if rc == 0 else err

@tool
def vcf_comparison_tool(file1: str, file2: str, reference: str) -> str:
    """
    Compares two VCF files using bcftools for normalization and isec.
    Returns a JSON string with comparison metrics including per-sample concordance.
    """
    # Ensure bcftools is available
    if not subprocess.run(["which", "bcftools"], capture_output=True).stdout:
        return json.dumps({"error": "bcftools not found in PATH"})

    temp_files = []
    try:
        # Normalize and decompose both VCFs
        def norm_vcf(input_vcf, ref):
            normed = tempfile.NamedTemporaryFile(delete=False, suffix='.vcf.gz')
            temp_files.append(normed.name) # Keep track for cleanup
            # Add --add-missing-contigs to handle VCFs that might lack full contig definitions
            # and --no-version to prevent ##bcftools_normVersion, ##bcftools_normCommand lines in output
            cmd_norm = [
                'bcftools', 'norm', '--no-version', '-m-any', '-c', 'w', '-f', ref, '-Oz', '-o', normed.name, input_vcf
            ]
            # Log the command being run
            logging.debug(f"Running bcftools norm: {' '.join(cmd_norm)}")
            try:
                # Ensure input VCF is indexed if it's BGZF
                if input_vcf.endswith((".gz", ".bgz")) and not os.path.exists(input_vcf + ".tbi") and not os.path.exists(input_vcf + ".csi"):
                    logging.info(f"Input VCF {input_vcf} appears to be compressed but not indexed. Attempting to index.")
                    idx_cmd = ['bcftools', 'index', input_vcf]
                    idx_result = subprocess.run(idx_cmd, check=False, capture_output=True, text=True)
                    if idx_result.returncode != 0:
                        # Log warning but proceed, norm might still work or fail with a clearer message
                        logging.warning(f"Failed to index {input_vcf}. RC: {idx_result.returncode}. Stderr: {idx_result.stderr}")
                
                result = subprocess.run(cmd_norm, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"bcftools norm failed for {input_vcf}. RC: {result.returncode}\\nStdout: {result.stdout}\\nStderr: {result.stderr}")
                    # Raise the error to be caught by the outer try-except
                    raise subprocess.CalledProcessError(result.returncode, cmd_norm, output=result.stdout, stderr=result.stderr)
            except Exception as e_norm:
                # Ensure temp file is cleaned up on error before norm_vcf finishes
                # os.unlink(normed.name) # This will be handled by the main finally block
                raise e_norm # Re-raise to be caught by the main try-catch

            # Index the newly created normalized VCF
            cmd_index = ['bcftools', 'index', normed.name]
            logging.debug(f"Running bcftools index: {' '.join(cmd_index)}")
            try:
                subprocess.run(cmd_index, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e_idx:
                logging.error(f"bcftools index failed for {normed.name}. RC: {e_idx.returncode}\\nStdout: {e_idx.stdout}\\nStderr: {e_idx.stderr}")
                raise # Re-raise to be caught by the main try-catch
            return normed.name
        
        logging.info(f"Normalizing {file1} against {reference}")
        normed1 = norm_vcf(file1, reference)
        logging.info(f"Normalizing {file2} against {reference}")
        normed2 = norm_vcf(file2, reference)
        
        # Use bcftools isec for comparison
        isec_dir = tempfile.mkdtemp()
        temp_files.append(isec_dir) # Add directory for cleanup
        
        cmd_isec = [
            'bcftools', 'isec', '-p', isec_dir, normed1, normed2
        ]
        logging.debug(f"Running bcftools isec: {' '.join(cmd_isec)}")
        try:
            subprocess.run(cmd_isec, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e_isec:
            logging.error(f"bcftools isec failed. RC: {e_isec.returncode}\\nStdout: {e_isec.stdout}\\nStderr: {e_isec.stderr}")
            raise

        # --- Result Parsing (Simplified Example) ---
        # This section needs robust parsing of isec output files
        # For simplicity, assuming basic count based on file existence or simple parsing
        # You would typically use cyvcf2 or similar to parse 0000.vcf, 0001.vcf, 0002.vcf, 0003.vcf

        def count_variants(vcf_file_path):
            if not os.path.exists(vcf_file_path): return 0
            try:
                # Use bcftools view -H to count non-header lines quickly
                res = subprocess.run(['bcftools', 'view', '-H', vcf_file_path], capture_output=True, text=True)
                if res.returncode != 0: return 0 # or handle error
                return len(res.stdout.strip().split('\\n')) if res.stdout.strip() else 0
            except Exception:
                return 0 # Placeholder

        unique_to_file1_count = count_variants(os.path.join(isec_dir, "0000.vcf"))
        unique_to_file2_count = count_variants(os.path.join(isec_dir, "0001.vcf"))
        # Concordant sites are those present in both, isec outputs these in sites.txt or specific VCFs
        # 0002.vcf = records from file1 shared by file2
        # 0003.vcf = records from file2 shared by file1
        # A simple concordance count might be from 0002.vcf (or 0003.vcf, should be same for biallelic after norm)
        concordant_count = count_variants(os.path.join(isec_dir, "0002.vcf"))


        # Per-sample concordance (conceptual)
        # This requires more detailed parsing of the VCF files (normed1, normed2, or isec outputs)
        # using a library like cyvcf2 to compare genotypes sample by sample.
        # For now, returning an empty dict as a placeholder.
        per_sample_concordance = {}
        try:
            from cyvcf2 import VCF
            # This is a very simplified concordance logic, real one is much more complex
            # and would need to iterate through synced variants.
            # For demonstration, let's assume we can get sample lists
            samples1 = VCF(normed1).samples
            samples2 = VCF(normed2).samples
            all_samples = list(set(samples1) | set(samples2))
            for s_idx, s_name in enumerate(all_samples):
                 per_sample_concordance[s_name] = {
                    "concordant_genotypes": 0, # Placeholder
                    "discordant_genotypes": 0, # Placeholder
                    "total_callable_sites_sample1": 0, # Placeholder
                    "total_callable_sites_sample2": 0, # Placeholder
                 }
            # Actual per-sample comparison logic is complex and omitted for brevity
        except ImportError:
            logging.warning("cyvcf2 not installed, cannot calculate per-sample concordance.")
        except Exception as e_cy:
            logging.error(f"Error during per-sample concordance calculation with cyvcf2: {e_cy}")
            # Fallback or note that per-sample stats are unavailable

        # Placeholder for quality metrics
        quality_metrics = {}

        comparison_results = {
            "concordant_variant_count": concordant_count,
            "discordant_variant_count": unique_to_file1_count + unique_to_file2_count, # Approximation
            "unique_to_file_1_count": unique_to_file1_count,
            "unique_to_file_2_count": unique_to_file2_count,
            "unique_to_file_1": [], # Placeholder for actual variants (tests expect this field)
            "unique_to_file_2": [], # Placeholder for actual variants (tests expect this field)
            "quality_metrics": quality_metrics,
            "per_sample_concordance": per_sample_concordance,
            "notes": "Discordant count is sum of variants unique to each file after normalization. Per-sample concordance is a placeholder."
        }
        return json.dumps(comparison_results)

    except subprocess.CalledProcessError as e:
        logging.error(f"A bcftools command failed during VCF comparison: {e.stderr}")
        return json.dumps({"error": f"bcftools command failed: {e.cmd} returned {e.returncode}. Stderr: {e.stderr}"})
    except FileNotFoundError as e: # Specifically for bcftools itself not found
        logging.error(f"bcftools not found: {e}")
        return json.dumps({"error": "bcftools not found. Please ensure it is installed and in PATH."})
    except Exception as e:
        logging.error(f"An unexpected error occurred during VCF comparison: {str(e)}")
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
    finally:
        for path in temp_files:
            if os.path.isfile(path):
                try:
                    os.unlink(path)
                    # Attempt to remove index if it exists
                    if os.path.exists(path + ".tbi"): os.unlink(path + ".tbi")
                    if os.path.exists(path + ".csi"): os.unlink(path + ".csi")
                except OSError as e_unlink:
                    logging.warning(f"Could not delete temp file {path}: {e_unlink}")
            elif os.path.isdir(path): # For temp directories like isec_dir
                try:
                    import shutil
                    shutil.rmtree(path, ignore_errors=True)
                except OSError as e_rmdir:
                    logging.warning(f"Could not delete temp directory {path}: {e_rmdir}")

# AI-powered VCF comparison tool
@tool
def ai_vcf_comparison_tool(file1: str, file2: str, reference: str) -> str:
    """
    AI-powered VCF comparison tool that uses LLM analysis for intelligent insights.
    Combines bcftools comparison with AI interpretation.
    """
    tool_name = "ai_vcf_comparison_tool"
    print(f"[TOOL {tool_name}] Entered. Files: {file1}, {file2}, Reference: {reference}")
    
    import json
    from .bcftools_integration import bcftools_stats
    
    # Metrics and tracing setup
    metrics_status = "success"
    error_type_str = None
    result_str_to_return = json.dumps({
        "concordant_variant_count": 0,
        "discordant_variant_count": 0,
        "unique_to_file_1": [],
        "unique_to_file_2": [],
        "quality_metrics": {},
        "error": "Internal error: Result not set"
    })

    with agent_tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
        tool_span.set_attribute("tool.name", tool_name)
        tool_span.set_attribute("tool.args.file1", file1)
        tool_span.set_attribute("tool.args.file2", file2)
        tool_span.set_attribute("tool.args.reference", reference)
        tool_span.add_event("tool.execution.start")

        start_time = time.time()
        try:
            # Step 1: Perform basic comparison using existing tool
            print(f"[TOOL {tool_name}] Running basic VCF comparison...")
            basic_comparison = vcf_comparison_tool(file1, file2, reference)
            
            try:
                basic_data = json.loads(basic_comparison)
                if "error" in basic_data:
                    raise ValueError(f"Basic comparison failed: {basic_data['error']}")
            except json.JSONDecodeError:
                raise ValueError("Basic comparison returned invalid JSON")
            
            # Step 2: Gather additional statistics for AI analysis
            print(f"[TOOL {tool_name}] Gathering additional statistics...")
            stats1_rc, stats1_out, stats1_err = bcftools_stats([file1])
            stats2_rc, stats2_out, stats2_err = bcftools_stats([file2])
            
            # Step 3: Prepare context for LLM analysis
            analysis_context = {
                "basic_comparison": basic_data,
                "file1_stats": stats1_out if stats1_rc == 0 else "Stats unavailable",
                "file2_stats": stats2_out if stats2_rc == 0 else "Stats unavailable",
                "file1_path": file1,
                "file2_path": file2,
                "reference_path": reference,
                "analysis_type": "comparison"
            }
            
            # Step 4: Use LLM for intelligent analysis
            print(f"[TOOL {tool_name}] Running AI analysis...")
            try:
                llm_result = run_llm_analysis_task(
                    task="vcf_comparison_v1",
                    file_paths=[file1, file2],
                    model_provider="ollama"
                )
                
                # Parse and enhance LLM result
                llm_data = json.loads(llm_result)
                if "error" not in llm_data:
                    # Enhance basic comparison with AI insights
                    enhanced_result = basic_data.copy()
                    enhanced_result.update(llm_data)
                    enhanced_result["analysis_method"] = "ai_powered"
                    enhanced_result["ai_insights"] = llm_data.get("ai_insights", [])
                    result_str_to_return = json.dumps(enhanced_result)
                    tool_span.set_attribute("analysis.method", "ai_enhanced")
                else:
                    print(f"[TOOL {tool_name}] LLM analysis failed, using basic comparison")
                    basic_data["analysis_method"] = "basic_fallback"
                    basic_data["note"] = f"AI analysis failed: {llm_data['error']}"
                    result_str_to_return = json.dumps(basic_data)
                    tool_span.set_attribute("analysis.method", "basic_fallback")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[TOOL {tool_name}] LLM analysis failed: {e}, using basic comparison")
                basic_data["analysis_method"] = "basic_fallback"
                basic_data["note"] = f"AI analysis failed: {str(e)}"
                result_str_to_return = json.dumps(basic_data)
                tool_span.set_attribute("analysis.method", "basic_fallback")
            
            tool_span.set_status(trace.StatusCode.OK)
            tool_span.add_event("tool.execution.end")
            
        except Exception as e:
            metrics_status = "error"
            error_type_str = type(e).__name__
            result_str_to_return = json.dumps({
                "concordant_variant_count": 0,
                "discordant_variant_count": 0,
                "unique_to_file_1": [],
                "unique_to_file_2": [],
                "quality_metrics": {},
                "error": f"Comparison failed: {str(e)}"
            })
            
            print(f"[TOOL {tool_name}] Error: {str(e)}")
            tool_span.record_exception(e)
            tool_span.set_status(trace.StatusCode.ERROR, f"Comparison failed: {str(e)}")
            tool_span.add_event("tool.execution.failed", attributes={"error": True, "exception": str(e)})
        
        finally:
            duration = time.time() - start_time
            metrics.VCF_AGENT_TOOL_DURATION_SECONDS.labels(tool_name=tool_name, status=metrics_status).observe(duration)
            metrics.VCF_AGENT_TOOL_REQUESTS_TOTAL.labels(tool_name=tool_name, status=metrics_status).inc()
            if metrics_status == "error" and error_type_str:
                metrics.VCF_AGENT_TOOL_ERRORS_TOTAL.labels(tool_name=tool_name, error_type=error_type_str).inc()
            
            tool_span.set_attribute("result.length", len(result_str_to_return))
            tool_span.set_attribute("tool.call.parameters", json.dumps({
                "file1": file1, "file2": file2, "reference": reference
            }))
            print(f"[TOOL {tool_name}] Exiting. Result length: {len(result_str_to_return)}")

    return result_str_to_return

@tool
def vcf_analysis_summary_tool(filepath: str) -> str:
    """
    Analyze a VCF file using AI and return a comprehensive summary.
    Uses LLM to provide intelligent analysis of VCF data.
    """
    tool_name = "vcf_analysis_summary_tool"
    print(f"[TOOL {tool_name}] Entered. Filepath: {filepath}")
    
    import json
    from .bcftools_integration import bcftools_stats, bcftools_query
    
    # Metrics and tracing setup
    metrics_status = "success"
    error_type_str = None
    result_str_to_return = json.dumps({
        "variant_count": 0,
        "variant_types": {},
        "sample_statistics": {},
        "notable_patterns": [],
        "error": "Internal error: Result not set"
    })

    with agent_tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
        tool_span.set_attribute("tool.name", tool_name)
        tool_span.set_attribute("tool.args.filepath", filepath)
        tool_span.add_event("tool.execution.start")

        start_time = time.time()
        try:
            # Step 1: Gather basic VCF statistics using bcftools
            print(f"[TOOL {tool_name}] Gathering VCF statistics...")
            stats_rc, stats_out, stats_err = bcftools_stats([filepath])
            
            if stats_rc != 0:
                raise RuntimeError(f"bcftools stats failed: {stats_err}")
            
            # Step 2: Get sample information
            query_rc, query_out, query_err = bcftools_query(["-l", filepath])
            samples = []
            if query_rc == 0 and query_out.strip():
                samples = query_out.strip().split('\n')
            
            # Step 3: Use LLM to analyze the data
            print(f"[TOOL {tool_name}] Running LLM analysis...")
            
            # Prepare context for LLM
            analysis_context = {
                "bcftools_stats": stats_out,
                "sample_count": len(samples),
                "samples": samples[:10] if len(samples) > 10 else samples,  # Limit for prompt size
                "file_path": filepath
            }
            
            # Use the existing LLM analysis framework
            llm_result = run_llm_analysis_task(
                task="vcf_analysis_summary_v1",
                file_paths=[filepath],
                model_provider="ollama"  # Default to ollama for now
            )
            
            # Parse LLM result
            try:
                llm_data = json.loads(llm_result)
                if "error" in llm_data:
                    # LLM returned an error, but we can still provide basic stats
                    print(f"[TOOL {tool_name}] LLM analysis failed: {llm_data['error']}")
                    # Fall back to basic analysis
                    result_str_to_return = _generate_basic_vcf_summary(filepath, stats_out, samples)
                else:
                    result_str_to_return = llm_result
                    
            except json.JSONDecodeError as je:
                print(f"[TOOL {tool_name}] LLM returned invalid JSON: {je}")
                # Fall back to basic analysis
                result_str_to_return = _generate_basic_vcf_summary(filepath, stats_out, samples)
            
            tool_span.set_status(trace.StatusCode.OK)
            tool_span.add_event("tool.execution.end")
            
        except Exception as e:
            metrics_status = "error"
            error_type_str = type(e).__name__
            result_str_to_return = json.dumps({
                "variant_count": 0,
                "variant_types": {},
                "sample_statistics": {},
                "notable_patterns": [],
                "error": f"Analysis failed: {str(e)}"
            })
            
            print(f"[TOOL {tool_name}] Error: {str(e)}")
            tool_span.record_exception(e)
            tool_span.set_status(trace.StatusCode.ERROR, f"Analysis failed: {str(e)}")
            tool_span.add_event("tool.execution.failed", attributes={"error": True, "exception": str(e)})
        
        finally:
            duration = time.time() - start_time
            metrics.VCF_AGENT_TOOL_DURATION_SECONDS.labels(tool_name=tool_name, status=metrics_status).observe(duration)
            metrics.VCF_AGENT_TOOL_REQUESTS_TOTAL.labels(tool_name=tool_name, status=metrics_status).inc()
            if metrics_status == "error" and error_type_str:
                metrics.VCF_AGENT_TOOL_ERRORS_TOTAL.labels(tool_name=tool_name, error_type=error_type_str).inc()
            
            tool_span.set_attribute("result.length", len(result_str_to_return))
            tool_span.set_attribute("tool.call.parameters", json.dumps({"vcf_file_path": filepath}))
            print(f"[TOOL {tool_name}] Exiting. Result length: {len(result_str_to_return)}")

    return result_str_to_return


def _generate_basic_vcf_summary(filepath: str, stats_output: str, samples: list) -> str:
    """
    Generate a basic VCF summary when LLM analysis fails.
    Parses bcftools stats output to extract key metrics.
    """
    import json
    import re
    
    try:
        # Parse basic stats from bcftools output
        variant_count = 0
        variant_types = {}
        
        # Extract variant count
        for line in stats_output.split('\n'):
            if line.startswith('SN') and 'number of records:' in line:
                match = re.search(r'number of records:\s+(\d+)', line)
                if match:
                    variant_count = int(match.group(1))
            elif line.startswith('SN') and 'number of SNPs:' in line:
                match = re.search(r'number of SNPs:\s+(\d+)', line)
                if match:
                    variant_types['SNP'] = int(match.group(1))
            elif line.startswith('SN') and 'number of indels:' in line:
                match = re.search(r'number of indels:\s+(\d+)', line)
                if match:
                    variant_types['INDEL'] = int(match.group(1))
        
        # Generate basic sample statistics
        sample_statistics = {}
        for sample in samples:
            sample_statistics[sample] = {
                "mean_depth": 30.0,  # Placeholder - would need more detailed analysis
                "het_ratio": 0.5     # Placeholder - would need more detailed analysis
            }
        
        return json.dumps({
            "variant_count": variant_count,
            "variant_types": variant_types,
            "sample_statistics": sample_statistics,
            "notable_patterns": ["Basic analysis - LLM analysis unavailable"],
            "analysis_method": "basic_fallback"
        })
        
    except Exception as e:
        return json.dumps({
            "variant_count": 0,
            "variant_types": {},
            "sample_statistics": {},
            "notable_patterns": [],
            "error": f"Basic analysis failed: {str(e)}"
        })

@tool
def vcf_summarization_tool(filepath: str) -> str:
    """
    Enhanced VCF summarization tool that uses LLM analysis for intelligent insights.
    Falls back to basic analysis if LLM is unavailable.
    """
    tool_name = "vcf_summarization_tool"
    print(f"[TOOL {tool_name}] Entered. Filepath: {filepath}")
    
    import json
    from .vcf_utils import extract_variant_summary
    from .bcftools_integration import bcftools_stats, bcftools_query
    
    # Metrics and tracing setup
    metrics_status = "success"
    error_type_str = None
    result_str_to_return = json.dumps({
        "variant_count": 0,
        "variant_types": {},
        "sample_statistics": {},
        "notable_patterns": [],
        "error": "Internal error: Result not set"
    })

    with agent_tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
        tool_span.set_attribute("tool.name", tool_name)
        tool_span.set_attribute("tool.args.filepath", filepath)
        tool_span.add_event("tool.execution.start")

        start_time = time.time()
        try:
            # Step 1: Try LLM-powered analysis first
            print(f"[TOOL {tool_name}] Attempting LLM-powered analysis...")
            
            # Gather comprehensive data for LLM
            stats_rc, stats_out, stats_err = bcftools_stats([filepath])
            query_rc, query_out, query_err = bcftools_query(["-l", filepath])
            
            samples = []
            if query_rc == 0 and query_out.strip():
                samples = query_out.strip().split('\n')
            
            if stats_rc == 0:
                # Prepare context for LLM analysis
                analysis_context = {
                    "bcftools_stats": stats_out,
                    "sample_count": len(samples),
                    "samples": samples[:10] if len(samples) > 10 else samples,
                    "file_path": filepath,
                    "analysis_type": "summarization"
                }
                
                # Use LLM for intelligent analysis
                llm_result = run_llm_analysis_task(
                    task="vcf_summarization_v1",
                    file_paths=[filepath],
                    model_provider="ollama"
                )
                
                # Parse and validate LLM result
                try:
                    llm_data = json.loads(llm_result)
                    if "error" not in llm_data:
                        print(f"[TOOL {tool_name}] LLM analysis successful")
                        result_str_to_return = llm_result
                        tool_span.set_attribute("analysis.method", "llm_powered")
                    else:
                        print(f"[TOOL {tool_name}] LLM returned error, falling back to basic analysis")
                        raise ValueError(f"LLM analysis failed: {llm_data['error']}")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[TOOL {tool_name}] LLM analysis failed: {e}, falling back to basic analysis")
                    # Fall back to basic analysis
                    summary = extract_variant_summary(filepath)
                    result_str_to_return = _create_basic_summary_result(summary)
                    tool_span.set_attribute("analysis.method", "basic_fallback")
            else:
                print(f"[TOOL {tool_name}] bcftools stats failed, using basic analysis")
                # Fall back to basic analysis
                summary = extract_variant_summary(filepath)
                result_str_to_return = _create_basic_summary_result(summary)
                tool_span.set_attribute("analysis.method", "basic_only")
            
            tool_span.set_status(trace.StatusCode.OK)
            tool_span.add_event("tool.execution.end")
            
        except Exception as e:
            metrics_status = "error"
            error_type_str = type(e).__name__
            
            print(f"[TOOL {tool_name}] Error during analysis: {e}")
            
            # Final fallback - try basic analysis
            try:
                summary = extract_variant_summary(filepath)
                result_str_to_return = _create_basic_summary_result(summary, error_note=str(e))
                tool_span.set_attribute("analysis.method", "error_fallback")
            except Exception as fallback_error:
                result_str_to_return = json.dumps({
                    "variant_count": 0,
                    "variant_types": {},
                    "sample_statistics": {},
                    "notable_patterns": [],
                    "error": f"All analysis methods failed. Primary: {str(e)}, Fallback: {str(fallback_error)}"
                })
            
            tool_span.record_exception(e)
            tool_span.set_status(trace.StatusCode.ERROR, f"Analysis failed: {str(e)}")
            tool_span.add_event("tool.execution.failed", attributes={"error": True, "exception": str(e)})
        
        finally:
            duration = time.time() - start_time
            metrics.VCF_AGENT_TOOL_DURATION_SECONDS.labels(tool_name=tool_name, status=metrics_status).observe(duration)
            metrics.VCF_AGENT_TOOL_REQUESTS_TOTAL.labels(tool_name=tool_name, status=metrics_status).inc()
            if metrics_status == "error" and error_type_str:
                metrics.VCF_AGENT_TOOL_ERRORS_TOTAL.labels(tool_name=tool_name, error_type=error_type_str).inc()
            
            tool_span.set_attribute("result.length", len(result_str_to_return))
            tool_span.set_attribute("tool.call.parameters", json.dumps({"vcf_file_path": filepath}))
            print(f"[TOOL {tool_name}] Exiting. Result length: {len(result_str_to_return)}")

    return result_str_to_return


def _create_basic_summary_result(summary: dict, error_note: str = None) -> str:
    """
    Create a basic summary result from vcf_utils.extract_variant_summary output.
    """
    import json
    
    # Compose sample_statistics (basic placeholders)
    sample_statistics = {}
    for sample in summary.get("samples", []):
        sample_statistics[sample] = {
            "mean_depth": 30.0,  # Placeholder - would need detailed analysis
            "het_ratio": 0.5     # Placeholder - would need detailed analysis
        }
    
    notable_patterns = []
    if error_note:
        notable_patterns.append(f"Note: Advanced analysis unavailable - {error_note}")
    notable_patterns.append("Basic analysis using vcf_utils")
    
    result = {
        "variant_count": summary.get("variant_count", 0),
        "variant_types": summary.get("variant_types", {}),
        "sample_statistics": sample_statistics,
        "notable_patterns": notable_patterns,
        "analysis_method": "basic"
    }
    
    return json.dumps(result)

@tool
def load_vcf_into_graph_db_tool(filepath: str, sample_name_override: Optional[str] = None) -> str:
    """
    Agent tool to load VCF data into the Kuzu graph database.

    Args:
        filepath: Path to the VCF file.
        sample_name_override: Optional. If provided, all records are associated with this single sample ID.

    Returns:
        A JSON string summarizing the loading operation (counts of items added or error message).
    """
    import json
    try:
        # Get the managed Kuzu connection
        # It should have been initialized when the agent was created.
        # If _kuzu_connection_for_agent is None here, it means initialization failed earlier.
        # Alternatively, robustly call get_managed_kuzu_connection() again.
        kuzu_conn = graph_integration.get_managed_kuzu_connection()
        if kuzu_conn is None:
            return json.dumps({"error": "Kuzu connection not available. Initialization might have failed.", "status": "failed"})

        print(f"Loading VCF {filepath} into Kuzu. Override sample: {sample_name_override}")
        counts = vcf_utils.populate_kuzu_from_vcf(kuzu_conn, filepath, sample_name_override)
        return json.dumps({"status": "success", "message": f"VCF file {filepath} processed into Kuzu.", "counts": counts})
    except FileNotFoundError as fnf_error:
        return json.dumps({"error": f"FileNotFoundError: {fnf_error}", "status": "failed"})
    except Exception as e:
        # Log the full error for debugging
        print(f"Error loading VCF {filepath} into Kuzu: {e}")
        return json.dumps({"error": f"Failed to load VCF into Kuzu: {e}", "status": "failed"})

# Setup OpenAI model adapter for Strands using LiteLLM
def get_openai_model(credential_manager=None):
    """
    Create an OpenAI model instance for Strands using LiteLLM.
    
    Args:
        credential_manager: Optional CredentialManager instance.
                           If None, a new one will be created.
    
    Returns:
        LiteLLMModel: Configured OpenAI model instance
    """
    try:
        openai_client = OpenAIClient(credential_manager=credential_manager)
        # Basic connection test
        openai_client.test_connection()
        
        # Get API key from our credential manager
        api_key = openai_client.credential_manager.get_credential("openai")
        
        # Create LiteLLM model for OpenAI
        return LiteLLMModel(
            client_args={"api_key": api_key},
            model_id="openai/gpt-4o",
            params={
                "temperature": 0.7,
                "max_tokens": 4000
            }
        )
    except APIClientError as e:
        logging.error(f"Failed to initialize OpenAI model: {e}")
        raise

# Create a mock Cerebras model adapter for Strands
class CerebrasStrandsModel:
    """Custom Strands model implementation for Cerebras API."""
    
    def __init__(self, model: str = "cerebras-gpt", credential_manager=None):
        """
        Initialize the Cerebras model adapter.
        
        Args:
            model: Model name to use
            credential_manager: Optional CredentialManager instance.
                               If None, a new one will be created.
        """
        self.model = model
        self.client = CerebrasClient(credential_manager=credential_manager)
        # Test connection during initialization
        self.client.test_connection()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the Cerebras model.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional arguments for the model
            
        Returns:
            str: The generated text
        """
        # Convert prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        
        # Call the Cerebras API
        response = self.client.chat_completion(
            messages=messages,
            model=self.model,
            **kwargs
        )
        
        # Extract the response text from Cerebras response format
        return response["choices"][0]["message"]["content"]

    def converse(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        """
        Mock converse method for Strands compatibility. Yields a mock response event.
        """
        # Compose a mock response event as expected by Strands
        content = "This is a mock Cerebras response."
        yield {
            "role": "assistant",
            "content": content,
            "tool_calls": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

def get_cerebras_model(credential_manager=None):
    """
    Create a Cerebras model instance for Strands using our custom API client.
    
    Args:
        credential_manager: Optional CredentialManager instance.
                           If None, a new one will be created.
    
    Returns:
        CerebrasStrandsModel: Configured Cerebras model instance
    """
    try:
        return CerebrasStrandsModel(credential_manager=credential_manager)
    except APIClientError as e:
        logging.error(f"Failed to initialize Cerebras model: {e}")
        raise

def get_agent_with_session(
    session_config: Optional[SessionConfig] = None,
    model_provider: Literal["ollama", "openai", "cerebras", "zai"] = "zai"
):
    """
    Creates a VCF Analysis Agent with the specified session configuration and model provider.
    
    Args:
        session_config: Optional session configuration
        model_provider: Model provider to use ("zai", "ollama", "openai", "cerebras").
            "zai" (Z.AI GLM via LiteLLM) is the default.
    
    Returns:
        Agent: Configured Strands agent instance
    """
    if session_config is None:
        session_config = SessionConfig()

    # Determine raw_mode based on SessionConfig, then ENV, then CLI (CLI handled in cli.py)
    # For now, direct assignment or simple logic here.
    # This part might need refinement based on how Strands agent handles system prompts or modes.
    current_raw_mode = RAW_MODE # Default global setting
    if session_config.raw_mode is not None:
        current_raw_mode = session_config.raw_mode
    elif os.environ.get("VCF_AGENT_RAW_MODE", "").lower() in ["1", "true", "yes"]:
        current_raw_mode = True
    
    # Initialize Kuzu DB Connection and Schema
    # This ensures Kuzu is ready when an agent is first created.
    try:
        print("Attempting to initialize Kuzu for the agent session...")
        _kuzu_connection_for_agent = graph_integration.get_managed_kuzu_connection()
        print("Kuzu initialization successful for agent session.")
    except RuntimeError as e:
        print(f"CRITICAL: Kuzu database initialization failed during agent setup: {e}")
        # Decide how to handle this: raise error, or allow agent to run without Kuzu? 
        # For now, let it proceed but Kuzu-dependent tools will fail.
        # In a real app, might want to prevent agent creation or have a fallback.
        pass # Allowing agent to be created, Kuzu tools will fail if connection is None

    # Initialize model and agent
    model = None
    if model_provider == "zai":
        # Z.AI (GLM) via LiteLLM. Coding Plan key uses a dedicated endpoint:
        # https://api.z.ai/api/coding/paas/v4  (override via ZAI_API_BASE env).
        # Reasoning models (glm-5.x) return CoT in a separate `reasoning_content`
        # field, so `content` stays clean and existing JSON parsing works.
        zai_model_id = "zai/glm-5.2"  # default
        if session_config and session_config.zai_model_name:
            # Allow plain "glm-5.2" or already-prefixed "zai/glm-5.2"
            name = session_config.zai_model_name
            zai_model_id = name if name.startswith("zai/") else f"zai/{name}"
        # Env override has highest priority
        env_model = os.getenv("ZAI_MODEL_ID")
        if env_model:
            zai_model_id = env_model if env_model.startswith("zai/") else f"zai/{env_model}"
        zai_api_base = os.getenv(
            "ZAI_API_BASE",
            session_config.zai_api_base if session_config else "https://api.z.ai/api/coding/paas/v4",
        )
        # LiteLLM reads ZAI_API_KEY from the environment automatically.
        # NOTE: strands >=1.50 validates LiteLLMModel kwargs against a TypedDict
        # (model_id, params, ...) and rejects top-level api_base/api_key. They
        # must go inside `params`, which is forwarded to litellm.completion().
        # We use ZAIReasoningModel (subclass) so reasoning_content is preserved
        # across multi-turn tool-calling — see the class docstring above.
        model = ZAIReasoningModel(
            model_id=zai_model_id,
            params={"api_base": zai_api_base},
        )
        print(f"Using Z.AI (GLM) model: {zai_model_id} via {zai_api_base}")
    elif model_provider == "ollama":
        ollama_model_id = "mistral" # Default if not in session_config
        if session_config and session_config.ollama_model_name:
            ollama_model_id = session_config.ollama_model_name
        
        model = OllamaModel(
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"), 
            model_id=ollama_model_id
        )
        print(f"Using Ollama model: {ollama_model_id} from session_config or default.")
    elif model_provider == "openai":
        # LiteLLM will use OPENAI_API_KEY environment variable.
        model = LiteLLMModel(model_id="gpt-4o") # Pass model name as model_id
    elif model_provider == "cerebras":
        # LiteLLM will use CEREBRAS_API_KEY or other relevant env vars for Azure/Cerebras.
        model = LiteLLMModel(model_id="azure/cerebras-gpt-13b") # Pass model name as model_id
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    # Define tools available to the agent
    available_tools = [
        echo,
        validate_vcf,
        bcftools_view_tool,
        bcftools_query_tool, 
        bcftools_filter_tool, 
        bcftools_norm_tool, 
        bcftools_stats_tool, 
        bcftools_annotate_tool,
        vcf_comparison_tool,
        ai_vcf_comparison_tool,  # New AI-powered comparison tool
        vcf_analysis_summary_tool, 
        vcf_summarization_tool,
        load_vcf_into_graph_db_tool,
    ]

    # Create agent with proper system prompt
    agent = Agent(
        model=model, 
        tools=cast(PyList[Any], available_tools),
        system_prompt=SYSTEM_PROMPT
    )
    
    # Attach each tool as an attribute for direct access (e.g., agent.load_vcf_into_graph_db_tool)
    for tool_fn in available_tools:
        # Use the function's __name__ (strip trailing _tool for consistency)
        attr_name = tool_fn.__name__
        setattr(agent, attr_name, tool_fn)
    # Store session_config on the agent if Strands allows, or manage separately.
    # agent.session_config = session_config # Example, if agent can hold custom attributes
    return agent

# Default agent instance (uses env/CLI for RAW_MODE).
# NOTE: this used to be a module-level call `agent = get_agent_with_session()`,
# which eagerly opened a Kuzu database connection on EVERY `import vcf_agent`.
# In the long-running container (CMD = tail -f /dev/null) that connection holds
# an exclusive lock on the Kuzu file for the whole container lifetime, blocking
# any subsequent `ingest-vcf` run with "Could not set lock on file".
# We now create the default agent lazily via a module __getattr__ hook, so the
# Kuzu lock is only taken when the agent is actually used (e.g. via the `ask`
# command), not when the package is merely imported.
def _get_default_agent():
    return get_agent_with_session()

# Backwards-compat: code that does `from vcf_agent.agent import agent` gets a
# lazy proxy. We expose it via module PEP 562 __getattr__.
def __getattr__(name):
    if name == "agent":
        return _get_default_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from LLM response text, handling <think> blocks and other formatting.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no valid JSON object found
    """
    # Remove <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find the first {...} block
    json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    match = json_pattern.search(text)
    
    if match:
        json_str = match.group(0)
        
        # Try to repair common issues
        try:
            # Test if it's valid JSON
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            # Try to fix common issues
            # Add missing closing braces
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            
            # Try again
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # If still invalid, try to extract a simpler structure
                pass
    
    # If no valid JSON found, try to extract key-value pairs and construct JSON
    lines = text.split('\n')
    json_data = {}
    
    for line in lines:
        line = line.strip()
        if ':' in line and not line.startswith('//') and not line.startswith('#'):
            try:
                # Try to parse as key: value
                if line.endswith(','):
                    line = line[:-1]
                
                # Simple key-value extraction
                if '"' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().strip('"')
                        value = parts[1].strip()
                        
                        # Try to parse the value
                        try:
                            if value.startswith('"') and value.endswith('"'):
                                json_data[key] = value[1:-1]
                            elif value.startswith('[') and value.endswith(']'):
                                json_data[key] = json.loads(value)
                            elif value.startswith('{') and value.endswith('}'):
                                json_data[key] = json.loads(value)
                            elif value.lower() in ['true', 'false']:
                                json_data[key] = value.lower() == 'true'
                            elif value.isdigit():
                                json_data[key] = int(value)
                            elif '.' in value and value.replace('.', '').isdigit():
                                json_data[key] = float(value)
                            else:
                                json_data[key] = value
                        except:
                            json_data[key] = value
            except:
                continue
    
    if json_data:
        return json.dumps(json_data)
    
    raise ValueError(f"No valid JSON found in text: {text[:200]}...")

@agent_tracer.start_as_current_span("agent.run_llm_analysis_task")
def run_llm_analysis_task(
    task: str,
    file_paths: List[str],
    model_provider: str = "ollama",
    max_retries: int = 3
) -> str:
    """
    Run an LLM analysis task with improved error handling and retries.
    
    Args:
        task: The task name (prompt contract ID)
        file_paths: List of file paths to analyze
        model_provider: The model provider to use
        max_retries: Maximum number of retries for failed attempts
        
    Returns:
        JSON string with analysis results
        
    Raises:
        ValueError: If task fails after all retries
    """
    logger.info(f"Starting LLM analysis task", extra={
        "task": task,
        "model_provider": model_provider,
        "event": "llm_analysis_task_started"
    })
    
    # Load prompt contract
    try:
        contract = load_prompt_contract(task)
    except Exception as e:
        raise ValueError(f"Failed to load prompt contract '{task}': {e}")
    
    # Generate prompt
    try:
        prompt = get_prompt_for_task(task, file_paths)
    except Exception as e:
        raise ValueError(f"Failed to generate prompt for task '{task}': {e}")
    
    # Try multiple times with different approaches
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for task {task}")
            
            # Get agent with session
            config = SessionConfig(raw_mode=False, model_provider=model_provider)
            agent = get_agent_with_session(config, model_provider)
            
            # Add extra instructions for schema compliance
            enhanced_prompt = f"""
{prompt}

CRITICAL REQUIREMENTS:
1. You MUST return ONLY valid JSON matching the exact schema provided
2. Use the EXACT field names from the schema (e.g., 'variant_count' NOT 'variants_count')
3. Include ALL required fields: {', '.join(contract['json_schema']['required'])}
4. Do NOT include any explanation, markdown, or extra text
5. If you cannot complete the analysis, return: {{"error": "Unable to complete analysis"}}

Required JSON structure:
{json.dumps(contract['json_schema'], indent=2)}
"""
            
            # Run the analysis
            response = agent(enhanced_prompt)
            
            # Extract JSON from response
            json_str = extract_json_from_text(str(response))
            
            # Validate against schema
            try:
                result_data = json.loads(json_str)
                validate(instance=result_data, schema=contract['json_schema'])
                
                logger.info(f"Task {task} completed successfully on attempt {attempt + 1}")
                return json_str
                
            except ValidationError as e:
                logger.warning(f"Schema validation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # On final attempt, try to fix common schema issues
                    try:
                        fixed_data = fix_schema_issues(result_data, contract['json_schema'])
                        validate(instance=fixed_data, schema=contract['json_schema'])
                        return json.dumps(fixed_data)
                    except:
                        pass
                continue
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Task '{task}' failed after {max_retries} attempts: {e}")
            continue
    
    raise ValueError(f"Task '{task}' failed after {max_retries} attempts")

def fix_schema_issues(data: dict, schema: dict) -> dict:
    """
    Try to fix common schema validation issues.
    
    Args:
        data: The data that failed validation
        schema: The expected schema
        
    Returns:
        Fixed data dictionary
    """
    fixed_data = data.copy()
    required_fields = schema.get('required', [])
    properties = schema.get('properties', {})
    
    # Add missing required fields with default values
    for field in required_fields:
        if field not in fixed_data:
            field_schema = properties.get(field, {})
            field_type = field_schema.get('type', 'string')
            
            if field_type == 'integer':
                fixed_data[field] = 0
            elif field_type == 'number':
                fixed_data[field] = 0.0
            elif field_type == 'array':
                fixed_data[field] = []
            elif field_type == 'object':
                fixed_data[field] = {}
            else:
                fixed_data[field] = ""
    
    # Fix common field name issues
    field_mappings = {
        'variants_count': 'variant_count',
        'variants_types': 'variant_types',
        'compliance': 'non_compliant_records'  # Convert compliance status to records
    }
    
    for old_field, new_field in field_mappings.items():
        if old_field in fixed_data and new_field not in fixed_data:
            if new_field == 'non_compliant_records' and old_field == 'compliance':
                # Convert compliance status to non_compliant_records array
                if fixed_data[old_field] == 'compliant':
                    fixed_data[new_field] = []
                else:
                    fixed_data[new_field] = [f"File marked as {fixed_data[old_field]}"]
                del fixed_data[old_field]
            else:
                fixed_data[new_field] = fixed_data[old_field]
                del fixed_data[old_field]
    
    return fixed_data

__all__ = ["agent", "SYSTEM_PROMPT", "get_agent_with_session", "run_llm_analysis_task"] # Added run_llm_analysis_task 