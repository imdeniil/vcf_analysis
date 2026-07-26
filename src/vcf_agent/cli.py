"""
VCF Analysis Agent CLI Entrypoint

Provides a comprehensive command-line interface for VCF analysis, genomic data management,
database operations, and compliance validation. Supports multiple AI model providers
(Ollama, OpenAI, Cerebras) and integrates with both LanceDB vector database and 
Kuzu graph database for advanced genomic data processing.

Interactive Commands:
• ask: Natural language interface to the VCF agent for analysis and queries

LanceDB Vector Database Operations:
• init-lancedb: Initialize LanceDB table for variant storage
• add-variant: Add individual variant records with embeddings
• search-embedding: Search variants by embedding vector similarity
• update-variant: Update existing variant records and metadata
• delete-variants: Remove variants based on SQL filter criteria
• create-lancedb-index: Create performance indexes on table columns
• filter-lancedb: Query variants by metadata using SQL expressions

VCF Processing & Database Population:
• ingest-vcf: Complete VCF file processing into both LanceDB and Kuzu databases
• populate-kuzu-from-vcf: Populate Kuzu graph database with VCF variant relationships

Compliance & Quality Validation:
• samspec: SAMspec compliance validation suite
  - validate: Single VCF file compliance validation
  - batch-validate: Multiple VCF file batch validation  
  - explain: Detailed violation explanations and remediation guidance

Global Options:
• --model: Select AI provider (ollama, openai, cerebras)
• --raw: Disable chain-of-thought reasoning for direct responses
• --credentials: Path to API credentials file
• --reference: Reference FASTA file for normalization

Common Usage Examples:
    # Interactive analysis
    $ python -m vcf_agent.cli ask "Analyze variants in this VCF file for pathogenic mutations"
    
    # Complete VCF processing pipeline
    $ python -m vcf_agent.cli ingest-vcf --vcf-file data.vcf.gz --batch-size 1000
    
    # LanceDB operations
    $ python -m vcf_agent.cli init-lancedb --db_path ./variants_db
    $ python -m vcf_agent.cli search-embedding --embedding 0.1,0.2,... --limit 5
    $ python -m vcf_agent.cli filter-lancedb --filter_sql "chrom = '1' AND pos > 1000000"
    
    # Compliance validation
    $ python -m vcf_agent.cli samspec validate file.vcf --format json --verbose
    $ python -m vcf_agent.cli samspec batch-validate *.vcf --output-dir reports/
    
    # Advanced operations
    $ python -m vcf_agent.cli create-lancedb-index --column chrom --index_type BTREE
    $ python -m vcf_agent.cli update-variant --variant_id rs123 --updates '{"clinical_significance": "Benign"}'

For detailed command help:
    $ python -m vcf_agent.cli <command> --help
"""

import argparse
import os
from typing import Literal, cast
import json
import sys
import time # For timing
from vcf_agent import metrics # Import metrics module

# OpenTelemetry Imports
from opentelemetry import trace
from .tracing import init_tracer, setup_auto_instrumentation

# Enhanced tracing imports - Phase 4.2 integration
from .enhanced_tracing import (
    get_enhanced_trace_service,
    vcf_operation_context,
    ai_provider_context
)

# Initialize OpenTelemetry Tracer for the CLI
# This should be done once, as early as possible.
# The service name helps distinguish traces from different components.
cli_tracer = init_tracer(service_name="vcf-agent-cli")
setup_auto_instrumentation() # Placeholder, will configure auto-instrumentation

# Initialize enhanced tracing service for CLI operations
cli_trace_service = get_enhanced_trace_service("vcf-agent-cli")

def main():
    """
    Command-line interface for the VCF Analysis Agent.

    Accepts a prompt string to invoke agent tools (e.g., validation, echo).
    Supports mock response for testing via the VCF_AGENT_CLI_MOCK_RESPONSE environment variable.
    Supports --raw / --no-think flag to disable chain-of-thought reasoning (raw output mode).
    Supports --model to select the model provider (ollama, openai, cerebras).
    Provides LanceDB integration commands.
    """
    mock_response = os.environ.get("VCF_AGENT_CLI_MOCK_RESPONSE")
    if mock_response is not None:
        print(mock_response)
        return
        
    parser = argparse.ArgumentParser(description="VCF Analysis Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    subparsers.required = False # Allow no subcommand to print help

    # Common arguments for agent interaction (can be added to main or specific subcommands)
    parser.add_argument(
        "--raw", "--no-think", action="store_true", 
        help="Disable chain-of-thought reasoning (raw output mode)"
    )
    parser.add_argument(
        "--model", type=str, choices=["zai", "ollama", "openai", "cerebras"], default="zai",
        help="Select model provider (default: zai = Z.AI GLM). "
             "Others: ollama (local), openai, cerebras."
    )
    parser.add_argument(
        "--credentials", type=str, 
        help="Path to JSON credentials file for API access"
    )
    parser.add_argument(
        "--reference", "--fasta", type=str, default=None,
        help="Path to reference FASTA for normalization (required for some tools)"
    )
    parser.add_argument(
        "--ollama-model", type=str, default=None, # Default is None, SessionConfig has its own default
        help="Specify the Ollama model name (e.g., qwen:4b, llama2). Overrides SessionConfig default."
    )

    # ADD 'ask' SUBCOMMAND
    ask_parser = subparsers.add_parser("ask", help="Ask the VCF agent a question or give it a command. This is the default if no other subcommand is given and text follows.")
    ask_parser.add_argument("prompt_text", type=str, help="The prompt text for the agent.")

    # LanceDB: init
    parser_init = subparsers.add_parser("init-lancedb", help="Initialize LanceDB table for variants")
    parser_init.add_argument("--db_path", type=str, default="./lancedb", help="Path to LanceDB directory")
    parser_init.add_argument("--table_name", type=str, default="variants", help="Table name")

    # LanceDB: add variant
    parser_add = subparsers.add_parser("add-variant", help="Add a variant record to LanceDB")
    parser_add.add_argument("--db_path", type=str, default="./lancedb")
    parser_add.add_argument("--table_name", type=str, default="variants")
    parser_add.add_argument("--variant_id", type=str, required=True)
    parser_add.add_argument("--chrom", type=str, required=True)
    parser_add.add_argument("--pos", type=int, required=True)
    parser_add.add_argument("--ref", type=str, required=True)
    parser_add.add_argument("--alt", type=str, required=True)
    parser_add.add_argument("--embedding", type=str, required=True, help="Comma-separated embedding vector")
    parser_add.add_argument("--clinical_significance", type=str, default=None)

    # LanceDB: search by embedding
    parser_search = subparsers.add_parser("search-embedding", help="Search variants by embedding vector")
    parser_search.add_argument("--db_path", type=str, default="./lancedb")
    parser_search.add_argument("--table_name", type=str, default="variants")
    parser_search.add_argument("--embedding", type=str, required=True, help="Comma-separated embedding vector")
    parser_search.add_argument("--limit", type=int, default=10)
    parser_search.add_argument("--filter_sql", type=str, default=None, help="Optional SQL filter string")

    # LanceDB: update variant
    parser_update = subparsers.add_parser("update-variant", help="Update a variant record in LanceDB")
    parser_update.add_argument("--db_path", type=str, default="./lancedb")
    parser_update.add_argument("--table_name", type=str, default="variants")
    parser_update.add_argument("--variant_id", type=str, required=True, help="ID of the variant to update")
    parser_update.add_argument("--updates", type=str, required=True, help="JSON string of updates, e.g., '{\"clinical_significance\": \"Benign\", \"pos\": 12346}'")

    # LanceDB: delete variants
    parser_delete = subparsers.add_parser("delete-variants", help="Delete variants from LanceDB based on a filter")
    parser_delete.add_argument("--db_path", type=str, default="./lancedb")
    parser_delete.add_argument("--table_name", type=str, default="variants")
    parser_delete.add_argument("--filter_sql", type=str, required=True, help="SQL filter string for deletion")

    # LanceDB: create scalar index
    parser_index = subparsers.add_parser("create-lancedb-index", help="Create a scalar index on a table column")
    parser_index.add_argument("--db_path", type=str, default="./lancedb")
    parser_index.add_argument("--table_name", type=str, default="variants")
    parser_index.add_argument("--column", type=str, required=True, help="Name of the column to index")
    parser_index.add_argument("--index_type", type=str, default=None, help="Optional: Type of index (e.g., BTREE, BITMAP)")
    parser_index.add_argument("--replace", action="store_true", help="Replace existing index if it exists")

    # LanceDB: filter variants by metadata
    parser_filter = subparsers.add_parser("filter-lancedb", help="Filter variants based on metadata using a SQL query.")
    parser_filter.add_argument("--db_path", type=str, default="./lancedb")
    parser_filter.add_argument("--table_name", type=str, default="variants")
    parser_filter.add_argument("--filter_sql", type=str, required=True, help="SQL filter string for metadata query (e.g., \"chrom = '1' AND pos > 1000\")")
    parser_filter.add_argument("--select_columns", type=str, default=None, help="Comma-separated list of columns to return (e.g., variant_id,chrom,pos)")
    parser_filter.add_argument("--limit", type=int, default=None, help="Maximum number of records to return")

    # VCF Ingestion: ingest-vcf
    parser_ingest = subparsers.add_parser("ingest-vcf", help="Ingest VCF file into both LanceDB and Kuzu databases")
    parser_ingest.add_argument("--vcf-file", type=str, required=True, help="Path to VCF file (.vcf or .vcf.gz)")
    parser_ingest.add_argument("--lancedb-path", type=str, default="./lancedb", help="LanceDB database path")
    parser_ingest.add_argument("--kuzu-path", type=str, default="./kuzu_db", help="Kuzu database path")
    parser_ingest.add_argument("--table-name", type=str, default="variants", help="LanceDB table name")
    parser_ingest.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing variants")
    parser_ingest.add_argument("--validate-only", action="store_true", help="Only validate VCF file, no ingestion")
    parser_ingest.add_argument("--resume-from", type=str, help="Resume from specific position (CHROM:POS)")
    parser_ingest.add_argument("--sample-name-override", type=str, help="Override sample name for single-sample VCFs")

    # Kuzu: populate from VCF
    parser_populate_kuzu = subparsers.add_parser("populate-kuzu-from-vcf", help="Populate Kuzu graph database from a VCF file.")
    parser_populate_kuzu.add_argument("--vcf_file_path", type=str, required=True, help="Path to the VCF file.")
    parser_populate_kuzu.add_argument("--kuzu_db_path", type=str, default="./kuzu_db", help="Path to Kuzu database directory.")
    # Optionally, add --sample_name_override if needed for the CLI too

    # SAMspec compliance validation
    parser_samspec = subparsers.add_parser("samspec", help="SAMspec compliance validation commands")
    samspec_subparsers = parser_samspec.add_subparsers(dest="samspec_command", help="SAMspec subcommands")
    
    # SAMspec validate
    samspec_validate = samspec_subparsers.add_parser("validate", help="Validate VCF file for SAMspec compliance")
    samspec_validate.add_argument("vcf_file", type=str, help="Path to VCF file to validate")
    samspec_validate.add_argument("--output", "-o", type=str, help="Output file for compliance report")
    samspec_validate.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    samspec_validate.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    samspec_validate.add_argument("--quiet", "-q", action="store_true", help="Only show summary")
    samspec_validate.add_argument("--verbose", "-v", action="store_true", help="Show detailed violation information")
    
    # SAMspec batch-validate
    samspec_batch = samspec_subparsers.add_parser("batch-validate", help="Validate multiple VCF files")
    samspec_batch.add_argument("vcf_files", nargs="+", type=str, help="Paths to VCF files to validate")
    samspec_batch.add_argument("--output-dir", "-d", type=str, help="Output directory for reports")
    samspec_batch.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    samspec_batch.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    samspec_batch.add_argument("--summary", action="store_true", help="Show summary report for all files")
    
    # SAMspec explain
    samspec_explain = samspec_subparsers.add_parser("explain", help="Explain SAMspec compliance violations")
    samspec_explain.add_argument("vcf_file", type=str, help="Path to VCF file to explain")
    samspec_explain.add_argument("--rule-id", type=str, help="Show details for specific rule ID")
    samspec_explain.add_argument("--level", choices=["critical", "warning", "info"], help="Filter violations by level")

    args = parser.parse_args()

    # If no command is given, but there are unparsed args, assume it's an 'ask' command
    # This makes 'ask' the default if non-option arguments are present.
    # We need to re-parse if this happens.
    if args.command is None and any(arg for arg in sys.argv[1:] if not arg.startswith('-')):
        # Check if the first non-option argument is a known command, if so, user forgot to specify it.
        # This is a basic check. More robust would be to inspect sys.argv directly.
        potential_command = next((arg for arg in sys.argv[1:] if not arg.startswith('-')), None)
        if potential_command not in subparsers.choices:
            # Re-parse with 'ask' prepended if it's not another known command
            # This assumes all non-flag args constitute the prompt
            prompt_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
            other_args = [arg for arg in sys.argv[1:] if arg.startswith('--')]
            
            # Filter out any already parsed known options from prompt_args
            # This is tricky because argparse has already consumed them.
            # A simpler approach for now: if no command, and sys.argv[1] isn't an option, it's a prompt.
            
            # Let's try a simpler default handling for now without re-parsing.
            # If args.command is None, and there was something that could be a prompt,
            # we'll manually set args.command to 'ask' and construct prompt_text.
            # This requires sys.argv inspection before parse_args, or a try-catch parse.

            # The following logic is based on the idea that if 'command' is None
            # AND there were positional arguments supplied that weren't consumed by options,
            # these positional arguments are intended for the 'ask' command.
            # This is hard to do robustly AFTER parse_args().
            # The `subparsers.required = False` and a check later is better.
            pass # Will handle default action later

    # LanceDB integration commands
    if args.command == "init-lancedb":
        with vcf_operation_context("cli_init_lancedb", args.db_path) as span:
            span.set_vcf_attributes(
                operation="database_initialization",
                db_path=args.db_path,
                table_name=args.table_name
            )
            
            try:
                from vcf_agent.lancedb_integration import get_db, get_or_create_table
                
                start_time = time.time()
                db = get_db(args.db_path)
                table = get_or_create_table(db, args.table_name)
                
                span.set_vcf_attributes(
                    success=True,
                    initialization_time_ms=(time.time() - start_time) * 1000
                )
                
                print(f"Initialized LanceDB table '{args.table_name}' at '{args.db_path}'")
                
            except Exception as e:
                span.set_vcf_attributes(
                    success=False,
                    error_message=str(e)
                )
                print(f"❌ Error initializing LanceDB: {e}", file=sys.stderr)
                sys.exit(1)
        return
    elif args.command == "add-variant":
        with vcf_operation_context("cli_add_variant", args.db_path) as span:
            span.set_vcf_attributes(
                operation="variant_addition",
                variant_id=args.variant_id,
                chromosome=args.chrom,
                position=args.pos,
                reference=args.ref,
                alternate=args.alt
            )
            
            try:
                from vcf_agent.lancedb_integration import get_db, get_or_create_table, add_variants
                
                start_time = time.time()
                
                # Parse embedding
                embedding = [float(x) for x in args.embedding.split(",")]
                span.set_vcf_attributes(embedding_dimensions=len(embedding))
                
                # Get database and table
                db = get_db(args.db_path)
                table = get_or_create_table(db, args.table_name)
                
                # Create variant record
                variant = {
                    "variant_id": args.variant_id,
                    "chrom": args.chrom,
                    "pos": args.pos,
                    "ref": args.ref,
                    "alt": args.alt,
                    "embedding": embedding,
                    "clinical_significance": args.clinical_significance,
                }
                
                # Add variant
                add_variants(table, [variant])
                
                span.set_vcf_attributes(
                    success=True,
                    addition_time_ms=(time.time() - start_time) * 1000,
                    variants_added=1,
                    has_clinical_significance=bool(args.clinical_significance)
                )
                
                print(f"Added variant {args.variant_id} to LanceDB.")
                
            except Exception as e:
                span.set_vcf_attributes(
                    success=False,
                    error_message=str(e)
                )
                print(f"❌ Error adding variant: {e}", file=sys.stderr)
                sys.exit(1)
        return
    elif args.command == "search-embedding":
        with vcf_operation_context("cli_search_embedding", args.db_path) as span:
            span.set_vcf_attributes(
                operation="embedding_search",
                limit=args.limit,
                has_filter=bool(args.filter_sql)
            )
            
            if args.filter_sql:
                # Mask sensitive data in filter for tracing
                from vcf_agent.lancedb_integration import mask_sensitive_sql
                masked_filter = mask_sensitive_sql(args.filter_sql)
                span.set_vcf_attributes(filter_sql_masked=masked_filter)
            
            try:
                from vcf_agent.lancedb_integration import get_db, get_or_create_table, search_by_embedding
                
                start_time = time.time()
                
                # Parse embedding
                embedding = [float(x) for x in args.embedding.split(",")]
                span.set_vcf_attributes(
                    query_embedding_dimensions=len(embedding),
                    embedding_parsed=True
                )
                
                # Get database and table
                db = get_db(args.db_path)
                table = get_or_create_table(db, args.table_name)
                
                # Perform search
                results = search_by_embedding(table, embedding, limit=args.limit, filter_sql=args.filter_sql)
                
                span.set_vcf_attributes(
                    success=True,
                    search_time_ms=(time.time() - start_time) * 1000,
                    results_found=len(results),
                    results_returned=min(len(results), args.limit)
                )
                
                print(results)
                
            except Exception as e:
                span.set_vcf_attributes(
                    success=False,
                    error_message=str(e)
                )
                print(f"❌ Error searching embeddings: {e}", file=sys.stderr)
                sys.exit(1)
        return
    elif args.command == "update-variant":
        from vcf_agent.lancedb_integration import get_db, get_or_create_table, update_variant
        db = get_db(args.db_path)
        table = get_or_create_table(db, args.table_name)
        try:
            updates = json.loads(args.updates)
            if not isinstance(updates, dict):
                raise ValueError("Updates must be a valid JSON object (dictionary).")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON string for updates: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        try:
            update_variant(table, args.variant_id, updates)
            print(f"Update command processed for variant {args.variant_id}.")
        except Exception as e:
            print(f"Error during variant update: {e}", file=sys.stderr)
            sys.exit(1)
        return
    elif args.command == "delete-variants":
        from vcf_agent.lancedb_integration import get_db, get_or_create_table, delete_variants
        db = get_db(args.db_path)
        table = get_or_create_table(db, args.table_name)
        delete_variants(table, args.filter_sql)
        print(f"Delete command processed with filter: {args.filter_sql}")
        return
    elif args.command == "create-lancedb-index":
        from vcf_agent.lancedb_integration import get_db, get_or_create_table, create_scalar_index
        db = get_db(args.db_path)
        table = get_or_create_table(db, args.table_name)
        # Basic kwargs handling for now, could be expanded if specific index params are common
        # For example, if num_partitions was a CLI arg, it would be passed here.
        index_kwargs = {}
        create_scalar_index(table, args.column, index_type=args.index_type, replace=args.replace, **index_kwargs)
        print(f"Index creation command processed for column '{args.column}' on table '{args.table_name}'.")
        return
    elif args.command == "filter-lancedb":
        from vcf_agent.lancedb_integration import get_db, get_or_create_table, filter_variants_by_metadata
        try:
            db = get_db(args.db_path)
            table = get_or_create_table(db, args.table_name)
            select_cols_list = args.select_columns.split(',') if args.select_columns else None
            results_df = filter_variants_by_metadata(table, args.filter_sql, select_columns=select_cols_list, limit=args.limit)
            print(f"Found {len(results_df)} variants matching filter:")
            print(results_df.to_string())
        except ValueError as ve:
            # Check if it's our specific unsafe SQL error
            if "Unsafe SQL filter detected" in str(ve):
                print(f"Error: {ve}", file=sys.stderr) # Ensure the exact message is printed
                sys.exit(1)
            else:
                raise # Re-raise other ValueErrors
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)            
        return
    elif args.command == "ingest-vcf":
        # Handle VCF ingestion into both LanceDB and Kuzu
        with cli_tracer.start_as_current_span("cli.ingest_vcf") as ingest_span:
            ingest_span.set_attribute("vcf.file_path", args.vcf_file)
            ingest_span.set_attribute("lancedb.path", args.lancedb_path)
            ingest_span.set_attribute("kuzu.path", args.kuzu_path)
            ingest_span.set_attribute("batch_size", args.batch_size)

            command_name = "ingest-vcf"
            status = "success"
            start_time = time.time()
            error_observed = None
            
            # CLI metrics
            cli_duration = metrics.CLI_COMMAND_DURATION_SECONDS
            cli_requests = metrics.CLI_COMMAND_REQUESTS_TOTAL
            cli_errors = metrics.CLI_COMMAND_ERRORS_TOTAL

            try:
                from vcf_agent.vcf_ingestion import VCFIngestionPipeline, IngestionConfig
                
                # Create ingestion configuration
                config = IngestionConfig(
                    vcf_file=args.vcf_file,
                    lancedb_path=args.lancedb_path,
                    kuzu_path=args.kuzu_path,
                    table_name=args.table_name,
                    batch_size=args.batch_size,
                    validate_only=args.validate_only,
                    resume_from=args.resume_from,
                    sample_name_override=args.sample_name_override
                )
                
                # Initialize and run ingestion pipeline
                pipeline = VCFIngestionPipeline(config)
                result = pipeline.execute()
                
                # Log success
                metrics.log.info(f"VCF ingestion completed successfully. "
                               f"Variants processed: {result.get('variants_processed', 0)}, "
                               f"LanceDB records: {result.get('lancedb_records', 0)}, "
                               f"Kuzu variants: {result.get('kuzu_variants', 0)}, "
                               f"Kuzu samples: {result.get('kuzu_samples', 0)}")
                
                ingest_span.set_attribute("variants_processed", result.get('variants_processed', 0))
                ingest_span.set_attribute("lancedb_records", result.get('lancedb_records', 0))
                ingest_span.set_attribute("kuzu_variants", result.get('kuzu_variants', 0))
                ingest_span.set_attribute("kuzu_samples", result.get('kuzu_samples', 0))
                
                print(f"✅ VCF ingestion completed successfully!")
                print(f"📊 Summary:")
                print(f"   • Variants processed: {result.get('variants_processed', 0)}")
                print(f"   • LanceDB records added: {result.get('lancedb_records', 0)}")
                print(f"   • Kuzu variants added: {result.get('kuzu_variants', 0)}")
                print(f"   • Kuzu samples added: {result.get('kuzu_samples', 0)}")

            except FileNotFoundError as e:
                status = "error"
                error_observed = e
                metrics.log.error(f"VCF file not found: {args.vcf_file}")
                ingest_span.record_exception(e)
                ingest_span.set_status(trace.StatusCode.ERROR, f"VCF file not found: {args.vcf_file}")
                print(f"❌ Error: VCF file not found at '{args.vcf_file}'", file=sys.stderr)
                
            except Exception as e:
                status = "error"
                error_observed = e
                metrics.log.error(f"Error during VCF ingestion: {e}")
                ingest_span.record_exception(e)
                ingest_span.set_status(trace.StatusCode.ERROR, "Error during VCF ingestion")
                print(f"❌ Error during VCF ingestion: {e}", file=sys.stderr)
                
            finally:
                duration = time.time() - start_time
                cli_duration.labels(command=command_name, status=status).observe(duration)
                cli_requests.labels(command=command_name, status=status).inc()
                
                metrics_to_push_list = [cli_duration, cli_requests]

                if status == "error" and error_observed:
                    error_type = type(error_observed).__name__
                    cli_errors.labels(command=command_name, error_type=error_type).inc()
                    metrics_to_push_list.append(cli_errors)
                
                metrics.push_job_metrics(job_name=command_name, metrics_to_push=metrics_to_push_list)

                if error_observed:
                    sys.exit(1)
        return
    elif args.command == "populate-kuzu-from-vcf":
        with cli_tracer.start_as_current_span("cli.populate_kuzu_from_vcf") as populate_span:
            populate_span.set_attribute("vcf.file_path", args.vcf_file_path)
            populate_span.set_attribute("kuzu.db_path", args.kuzu_db_path)

            from vcf_agent.graph_integration import get_kuzu_db_connection, create_schema
            from vcf_agent.vcf_utils import populate_kuzu_from_vcf
            import gc

            command_name = "populate-kuzu-from-vcf"
            status = "success"
            start_time = time.time()
            error_observed = None
            # Ensure CLI metrics are defined; they are not auto-registered to http_registry in metrics.py
            # For push, we collect their current state.
            cli_duration = metrics.CLI_COMMAND_DURATION_SECONDS
            cli_requests = metrics.CLI_COMMAND_REQUESTS_TOTAL
            cli_errors = metrics.CLI_COMMAND_ERRORS_TOTAL

            try:
                kuzu_conn = None # Define kuzu_conn here to ensure it's in scope for finally
                try:
                    kuzu_conn = get_kuzu_db_connection(db_path=args.kuzu_db_path)
                    create_schema(kuzu_conn) # Ensure schema exists
                    
                    metrics.log.info(f"Populating Kuzu graph at '{args.kuzu_db_path}' from VCF file '{args.vcf_file_path}'...")
                    counts = populate_kuzu_from_vcf(kuzu_conn=kuzu_conn, vcf_path=args.vcf_file_path)
                    metrics.log.info(f"Successfully populated Kuzu graph from VCF. Variants added: {counts.get('variants', 0)}, Samples added/found: {counts.get('samples', 0)}, Links created: {counts.get('links', 0)}. ")
                    populate_span.set_attribute("kuzu.variants_added", counts.get('variants', 0))
                    populate_span.set_attribute("kuzu.samples_added_found", counts.get('samples', 0))
                    populate_span.set_attribute("kuzu.links_created", counts.get('links', 0))

                except FileNotFoundError as e:
                    status = "error"
                    error_observed = e
                    metrics.log.error(f"Error: VCF file not found at '{args.vcf_file_path}'. Details: {e}", file=sys.stderr)
                    populate_span.record_exception(e)
                    populate_span.set_status(trace.StatusCode.ERROR, f"VCF file not found: {args.vcf_file_path}")
                except Exception as e:
                    status = "error"
                    error_observed = e
                    metrics.log.error(f"Error during Kuzu population: {e}", file=sys.stderr)
                    populate_span.record_exception(e)
                    populate_span.set_status(trace.StatusCode.ERROR, "Error during Kuzu population")
                finally:
                    if kuzu_conn:
                        del kuzu_conn 
                    gc.collect() 
            finally:
                duration = time.time() - start_time
                cli_duration.labels(command=command_name, status=status).observe(duration)
                cli_requests.labels(command=command_name, status=status).inc()
                
                metrics_to_push_list = [cli_duration, cli_requests]

                if status == "error" and error_observed:
                    error_type = type(error_observed).__name__
                    cli_errors.labels(command=command_name, error_type=error_type).inc()
                    metrics_to_push_list.append(cli_errors)
                
                metrics.push_job_metrics(job_name=command_name, metrics_to_push=metrics_to_push_list)

                if error_observed: # Re-raise or exit after pushing metrics
                    sys.exit(1)
        return
    elif args.command == "samspec":
        # Handle SAMspec compliance validation commands
        from vcf_agent.samspec_compliance import validate_vcf_samspec_compliance
        from pathlib import Path
        import json
        
        def format_text_report(report, verbose=False, quiet=False):
            """Format compliance report as text."""
            lines = []
            
            if not quiet:
                lines.append("SAMspec Compliance Report")
                lines.append("=" * 50)
                lines.append(f"File: {report.file_path}")
                lines.append(f"VCF Version: {report.vcf_version or 'Unknown'}")
                lines.append(f"Compliant: {'Yes' if report.is_compliant else 'No'}")
                lines.append("")
            
            # Summary
            lines.append("Violation Summary:")
            lines.append(f"  Critical: {report.critical_count}")
            lines.append(f"  Warnings: {report.warning_count}")
            lines.append(f"  Info: {report.info_count}")
            lines.append(f"  Total: {report.total_violations}")
            
            # Detailed violations
            if verbose and report.violations:
                lines.append("\nDetailed Violations:")
                lines.append("-" * 30)
                
                for i, violation in enumerate(report.violations, 1):
                    lines.append(f"\n{i}. {violation.rule_id} ({violation.level.value.upper()})")
                    lines.append(f"   {violation.message}")
                    
                    if violation.line_number:
                        lines.append(f"   Line: {violation.line_number}")
                    if violation.field:
                        lines.append(f"   Field: {violation.field}")
                    if violation.suggestion:
                        lines.append(f"   Suggestion: {violation.suggestion}")
            
            return "\n".join(lines)
        
        def format_summary_report(reports, strict=False):
            """Format summary report for multiple files."""
            lines = []
            lines.append("SAMspec Compliance Summary Report")
            lines.append("=" * 50)
            
            total_files = len(reports)
            compliant_files = sum(1 for r in reports if r.is_compliant)
            failed_files = total_files - compliant_files
            
            if strict:
                # In strict mode, warnings count as failures
                failed_files = sum(1 for r in reports if r.critical_count > 0 or r.warning_count > 0)
                compliant_files = total_files - failed_files
            
            lines.append(f"Total Files: {total_files}")
            lines.append(f"Compliant: {compliant_files}")
            lines.append(f"Non-compliant: {failed_files}")
            lines.append(f"Strict Mode: {'Yes' if strict else 'No'}")
            lines.append("")
            
            # Per-file summary
            lines.append("Per-File Results:")
            lines.append("-" * 30)
            
            for report in reports:
                file_name = Path(report.file_path).name
                status = "PASS"
                
                if report.critical_count > 0:
                    status = "FAIL"
                elif strict and report.warning_count > 0:
                    status = "FAIL"
                
                lines.append(f"{file_name}: {status} "
                            f"(C:{report.critical_count}, W:{report.warning_count}, I:{report.info_count})")
            
            # Overall statistics
            total_critical = sum(r.critical_count for r in reports)
            total_warnings = sum(r.warning_count for r in reports)
            total_info = sum(r.info_count for r in reports)
            
            lines.append("")
            lines.append("Overall Statistics:")
            lines.append(f"  Total Critical Violations: {total_critical}")
            lines.append(f"  Total Warnings: {total_warnings}")
            lines.append(f"  Total Info: {total_info}")
            
            return "\n".join(lines)
        
        if args.samspec_command == "validate":
            try:
                # Perform validation
                print(f"Validating VCF file: {args.vcf_file}")
                report = validate_vcf_samspec_compliance(args.vcf_file)
                
                # Generate output
                if args.format == 'json':
                    output_content = json.dumps(report.to_dict(), indent=2)
                else:
                    output_content = format_text_report(report, args.verbose, args.quiet)
                
                # Write output
                if args.output:
                    Path(args.output).write_text(output_content)
                    print(f"Report written to: {args.output}")
                else:
                    print(output_content)
                
                # Determine exit code
                exit_code = 0
                if report.critical_count > 0:
                    exit_code = 1
                elif args.strict and report.warning_count > 0:
                    exit_code = 1
                
                if exit_code != 0:
                    print(f"\nValidation failed with {report.critical_count} critical "
                          f"and {report.warning_count} warning violations.", file=sys.stderr)
                else:
                    print(f"\nValidation passed! File is SAMspec compliant.")
                
                sys.exit(exit_code)
                
            except Exception as e:
                print(f"Error during validation: {str(e)}", file=sys.stderr)
                sys.exit(1)
        
        elif args.samspec_command == "batch-validate":
            try:
                results = []
                failed_files = []
                
                # Create output directory if specified
                if args.output_dir:
                    output_path = Path(args.output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                
                # Validate each file
                for vcf_file in args.vcf_files:
                    print(f"Validating: {vcf_file}")
                    
                    try:
                        report = validate_vcf_samspec_compliance(vcf_file)
                        results.append(report)
                        
                        # Check if file failed validation
                        file_failed = report.critical_count > 0
                        if args.strict and report.warning_count > 0:
                            file_failed = True
                        
                        if file_failed:
                            failed_files.append(vcf_file)
                        
                        # Generate individual report if output directory specified
                        if args.output_dir:
                            file_stem = Path(vcf_file).stem
                            if args.format == 'json':
                                report_file = output_path / f"{file_stem}_compliance.json"
                                content = json.dumps(report.to_dict(), indent=2)
                            else:
                                report_file = output_path / f"{file_stem}_compliance.txt"
                                content = format_text_report(report, verbose=True, quiet=False)
                            
                            report_file.write_text(content)
                            print(f"  Report: {report_file}")
                        
                        # Show brief status
                        status = "PASS" if not file_failed else "FAIL"
                        print(f"  Status: {status} ({report.critical_count} critical, "
                              f"{report.warning_count} warnings)")
                        
                    except Exception as e:
                        print(f"  Error: {str(e)}", file=sys.stderr)
                        failed_files.append(vcf_file)
                
                # Generate summary report
                if args.summary and results:
                    summary_content = format_summary_report(results, args.strict)
                    
                    if args.output_dir:
                        summary_file = output_path / f"compliance_summary.{args.format}"
                        if args.format == 'json':
                            summary_data = {
                                "total_files": len(results),
                                "passed_files": len(results) - len(failed_files),
                                "failed_files": len(failed_files),
                                "strict_mode": args.strict,
                                "files": [report.to_dict() for report in results]
                            }
                            summary_file.write_text(json.dumps(summary_data, indent=2))
                        else:
                            summary_file.write_text(summary_content)
                        print(f"\nSummary report: {summary_file}")
                    else:
                        print("\n" + summary_content)
                
                # Final status
                total_files = len(args.vcf_files)
                passed_files = total_files - len(failed_files)
                
                print(f"\nBatch validation complete:")
                print(f"  Total files: {total_files}")
                print(f"  Passed: {passed_files}")
                print(f"  Failed: {len(failed_files)}")
                
                if failed_files:
                    print(f"\nFailed files:")
                    for file in failed_files:
                        print(f"  - {file}")
                    sys.exit(1)
                else:
                    print(f"\nAll files passed SAMspec compliance validation!")
                    sys.exit(0)
                    
            except Exception as e:
                print(f"Error during batch validation: {str(e)}", file=sys.stderr)
                sys.exit(1)
        
        elif args.samspec_command == "explain":
            try:
                report = validate_vcf_samspec_compliance(args.vcf_file)
                
                # Filter violations if requested
                violations = report.violations
                if args.rule_id:
                    violations = [v for v in violations if v.rule_id == args.rule_id]
                if args.level:
                    violations = [v for v in violations if v.level.value == args.level]
                
                if not violations:
                    if args.rule_id or args.level:
                        print("No violations found matching the specified criteria.")
                    else:
                        print("No compliance violations found. File is SAMspec compliant!")
                    return
                
                print(f"SAMspec Compliance Violations for: {args.vcf_file}")
                print("=" * 60)
                
                for i, violation in enumerate(violations, 1):
                    print(f"\n{i}. {violation.rule_id} ({violation.level.value.upper()})")
                    print(f"   Message: {violation.message}")
                    
                    if violation.line_number:
                        print(f"   Line: {violation.line_number}")
                    if violation.field:
                        print(f"   Field: {violation.field}")
                    if violation.value:
                        print(f"   Value: {violation.value}")
                    if violation.suggestion:
                        print(f"   Suggestion: {violation.suggestion}")
                
                print(f"\nTotal violations: {len(violations)}")
                
            except Exception as e:
                print(f"Error during explanation: {str(e)}", file=sys.stderr)
                sys.exit(1)
        
        return

    # Agent interaction logic (now specifically for 'ask' or default)
    # This block should be executed if args.command == "ask"
    # OR if args.command is None (meaning no other subcommand was explicitly called)
    
    prompt_to_run = None
    if args.command == "ask":
        prompt_to_run = args.prompt_text
    elif args.command is None:
        # This is the tricky part: how to get the prompt if no subcommand was specified?
        # parse_args() with nargs='?' for a positional argument on main parser
        # normally handles this.
        # For now, if command is None, we assume the user wanted help or made an error.
        # The 'ask' command must be explicit.
        parser.print_help()
        sys.exit(0) # Exit cleanly after help

    # This means args.command MUST be "ask" to reach here
    # (or other future agent-related subcommands)
    
    if prompt_to_run:
        if args.raw:
            os.environ["VCF_AGENT_RAW_MODE"] = "1"

        from vcf_agent.agent import get_agent_with_session
        from vcf_agent.config import SessionConfig
        
        model_provider = cast(Literal["zai", "ollama", "openai", "cerebras"], args.model)
        session_config = SessionConfig(
            raw_mode=args.raw if args.raw else None,
            model_provider=model_provider,
            credentials_file=args.credentials,
            reference_fasta=args.reference,
            ollama_model_name=args.ollama_model if args.ollama_model else None, # Pass it here
            zai_model_name=os.getenv("ZAI_MODEL_ID", "glm-5.2"),  # Z.AI GLM model override via env
        )
        
        agent_instance = get_agent_with_session( # Renamed to avoid conflict
            session_config=session_config,
            model_provider=model_provider
        )
        
        # The actual agent call within a span
        with cli_tracer.start_as_current_span("cli.agent_ask_interaction") as prompt_span: # Renamed span
            prompt_span.set_attribute("agent.prompt", prompt_to_run)
            prompt_span.set_attribute("agent.model_provider", model_provider)
            prompt_span.set_attribute("agent.raw_mode", args.raw if args.raw else False)
            
            ai_interaction_start_time = time.time()
            ai_status_metric = "success"
            ai_error_type_for_metric = None # Placeholder for actual error type if one occurs

            try:
                raw_agent_response = agent_instance(prompt_to_run)
                
                output_text = ""
                if isinstance(raw_agent_response, str):
                    output_text = raw_agent_response
                elif hasattr(raw_agent_response, 'response') and isinstance(getattr(raw_agent_response, 'response'), str):
                    output_text = getattr(raw_agent_response, 'response')
                elif hasattr(raw_agent_response, 'text') and isinstance(getattr(raw_agent_response, 'text'), str):
                    # Common alternative attribute name for response text
                    output_text = getattr(raw_agent_response, 'text')
                else:
                    output_text = str(raw_agent_response) 
                
                prompt_span.set_attribute("agent.response_length", len(output_text) if output_text is not None else 0)
                prompt_span.set_status(trace.StatusCode.OK)
                print(output_text)
            except Exception as e:
                ai_status_metric = "error"
                ai_error_type_for_metric = type(e).__name__
                prompt_span.record_exception(e)
                prompt_span.set_status(trace.StatusCode.ERROR, str(e))
                print(f"Error during agent execution: {e}", file=sys.stderr)
                # Re-raise or handle as appropriate; for now, we'll let it propagate if it's critical
                # For metrics, we capture the error. If we sys.exit(1) here, finally might not run as expected in all cases.
                # However, the original code did sys.exit(1)
                raise # Re-raising to ensure original exit behavior if this was unhandled.
            finally:
                ai_duration = time.time() - ai_interaction_start_time
                # Using "cli_ask" as a generic endpoint_task for metrics from this CLI interaction.
                # Token counts are not readily available here without deeper parsing of Strands agent internals.
                metrics.VCF_AGENT_AI_RESPONSE_SECONDS.labels(
                    model_provider=model_provider, 
                    endpoint_task="cli_ask", # Generic task for CLI 'ask'
                    status=ai_status_metric
                ).observe(ai_duration)
                metrics.VCF_AGENT_AI_REQUESTS_TOTAL.labels(
                    model_provider=model_provider, 
                    endpoint_task="cli_ask", 
                    status=ai_status_metric 
                ).inc()
                if ai_status_metric == "error" and ai_error_type_for_metric:
                    metrics.VCF_AGENT_AI_ERRORS_TOTAL.labels(
                        model_provider=model_provider, 
                        endpoint_task="cli_ask", 
                        error_type=ai_error_type_for_metric
                    ).inc()
        
        # Metrics server start - consider if this should only be for 'ask'
        metrics.start_metrics_http_server() 

    # Metrics server start - consider if this should only be for 'ask'
    # Note: Removed 60-second sleep for testing purposes

    # Ensure OTel spans are flushed before exit (this logic might already be present or adapted)
    # Based on previous versions, the OTel flush was here.
    current_provider = trace.get_tracer_provider()
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider # Ensure import if not global
    if isinstance(current_provider, SdkTracerProvider):
        print("\n[CLI] Forcing flush of OTel spans...")
        try:
            current_provider.force_flush(timeout_millis=5000) # 5 second timeout
            print("[CLI] OTel spans flushed.")
        except Exception as e:
            print(f"[CLI] Error during OTel span flush: {e}")
    else:
        print("\n[CLI] No SDK TracerProvider found, skipping flush.")

if __name__ == "__main__":
    main() 